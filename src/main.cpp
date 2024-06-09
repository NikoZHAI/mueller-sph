#define GL_SILENCE_DEPRECATION
#if __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#include <GL/freeglut.h>
#endif

#include <atomic>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>
#include <thread>
#include <cstddef>
#include <mutex>
#include <condition_variable>
#include <pthread.h>
#include <chrono>
#include <map>
using namespace std;

#include <eigen3/Eigen/Dense>
using namespace Eigen;

// "Particle-Based Fluid Simulation for Interactive Applications" by Müller et al.
// solver parameters
const static Vector2f G(0.f, -10.f);   // external (gravitational) forces
const static float REST_DENS = 300.f;  // rest density
const static float GAS_CONST = 2000.f; // const for equation of state
const static float H = 16.f;		   // kernel radius
const static float HSQ = H * H;		   // radius^2 for optimization
const static float MASS = 2.5f;		   // assume all particles have the same mass
const static float VISC = 200.f;	   // viscosity constant
const static float DT = 0.0007f;	   // integration timestep

// smoothing kernels defined in Müller and their gradients
// adapted to 2D per "SPH Based Shallow Water Simulation" by Solenthaler et al.
const static float POLY6 = 4.f / (M_PI * pow(H, 8.f));
const static float SPIKY_GRAD = -10.f / (M_PI * pow(H, 5.f));
const static float VISC_LAP = 40.f / (M_PI * pow(H, 5.f));

// simulation parameters
const static float EPS = H; // boundary epsilon
const static float BOUND_DAMPING = -0.5f;

// particle data structure
// stores position, velocity, and force for integration
// stores density (rho) and pressure values for SPH
struct Particle
{
	Particle(float _x, float _y) : x(_x, _y), v(0.f, 0.f), f(0.f, 0.f), rho(0), p(0.f) {}
	Vector2f x, v, f;
	float rho, p;

	bool operator< (const Particle &rhs) const
	{ return x(0) < rhs.x(0) || (x(0) == rhs.x(0) && x(1) < rhs.x(1)); }
};

// solver data
static vector<Particle> particles;

// interaction
const static int MAX_PARTICLES = 105000;
const static int DAM_PARTICLES = 5000;
const static int BLOCK_PARTICLES = 5000;

// rendering projection parameters
const static int WINDOW_WIDTH = 800;
const static int WINDOW_HEIGHT = 600;
const static double VIEW_WIDTH = 1.5 * 800.f;
const static double VIEW_HEIGHT = 1.5 * 600.f;


static std::mutex log_mutex;
#define log_info(arg) do { \
	std::unique_lock<std::mutex> local(log_mutex); \
	std::cout << arg << std::endl; \
} while(0)

#define log_info_rewind(arg) do { \
	std::unique_lock<std::mutex> local(log_mutex); \
	std::cout << arg << std::flush; \
} while(0)

static constexpr auto red_start("\033[0;31m");
static constexpr auto red_end("\033[0m");
#define log_warn(arg) do { \
	std::unique_lock<std::mutex> local(log_mutex); \
	std::cout << red_start << arg << red_end << std::endl; \
} while(0)

template<class T, std::size_t Capacity>
class ThreadSafeRingBuffer
{
    using Self              = ThreadSafeRingBuffer<T, Capacity>;
public:
    ////////////////////////////////////////////////////////////////////////////////
    // Traits
    //
    // Traits of the ThreadSafeRingBuffer class
    //
    using value_type        = T;
    using pointer           = value_type*;
    using reference         = value_type&;
    using const_pointer     = const value_type*;
    using const_reference   = const value_type&;
    using size_type         = std::size_t;
    using mutex_type        = std::mutex;
    using boolean           = bool;

    static constexpr size_type capacity()
    { return Capacity; }


public:
    ////////////////////////////////////////////////////////////////////////////////
    // Public Member Functions
    //
    //
    //
    size_type size()
    {
        std::lock_guard<std::mutex> lock(m_lock);
        size_type s = m_tail-m_head;
        return s;
    }


    /**
     * @brief Push `item` into the buffer
     *
     * @return `true` if success, `false` if queue full
    */
    boolean push_back( const_reference item )
    {
        bool success = false;
        m_lock.lock();
        size_type next = (m_head + 1ul) % capacity();
        if ( next != m_tail )
        {
            m_data[m_head] = item;
            m_head = next;
            success = true;
        }
        m_lock.unlock();
        return success;
    }


    /**
     * @brief Push `item` into the buffer
     *
     * @return `true` if success, `false` if queue full
    */
    boolean push_back( value_type&& item )
    {
        bool success = false;
        m_lock.lock();
        size_type next = (m_head + 1ul) % capacity();
        if ( next != m_tail )
        {
            m_data[m_head] = std::move(item);
            m_head = next;
            success = true;
        }
        m_lock.unlock();
        return success;
    }


    /**
     * @brief Get `item` from the buffer
     *
     * @return `true` if success, `false` if queue empty
    */
    boolean pop_front(reference &item)
    {
        bool success = false;
        m_lock.lock();
        if ( m_tail != m_head )
        {
            item = m_data[m_tail];
            m_tail = (m_tail + 1ul) % capacity();
            success = true;
        }
        m_lock.unlock();
        return success;
    }


    /**
     * @brief Clear the buffer
    */
    void clear()
    {
        m_lock.lock();
        m_tail = m_head;
        m_lock.unlock();
    }


    // assert stackoverflow
    ThreadSafeRingBuffer()
    { static_assert( sizeof( decltype(m_data) ) <= 8192UL ); }

    // disable default copy/move constructions
    ThreadSafeRingBuffer( const Self& ) = delete;


private:
    value_type              m_data[Capacity]    = {};
    size_type               m_head              = 0ul;
    size_type               m_tail              = 0ul;
    mutex_type              m_lock;
};


struct JobDescriptor
{
using Function = std::function<void(const JobDescriptor *)>;

public:
	int id;
	int start;
	int end;
	Function func;

	void run(const JobDescriptor *job) const { this->func(job); }
};


static unsigned nproc = std::thread::hardware_concurrency();
static std::mutex wakeMtx;
static std::condition_variable wakeCv;
static bool stopped;
static ThreadSafeRingBuffer<JobDescriptor*, 256> jobs;
static std::atomic_ulong sem;
static std::atomic_int threadCount;
static std::vector<JobDescriptor> descs;

static void threadWorker()
{
	JobDescriptor *job;
	int tid = threadCount.fetch_add(1);
	std::stringstream ss;
	ss << "sph-worker-" << tid;
	log_info(ss.str().c_str());
	if (pthread_setname_np(pthread_self(), ss.str().c_str()))
		log_warn( "tid=" << tid << " pthread_setname_np failed!");

	log_info("thread " << tid << " started");

	while (!stopped) {
		if (jobs.pop_front(job)) {
			job->run(job);
			--sem;
		} else {
			std::unique_lock<std::mutex> lock(wakeMtx);
			wakeCv.wait(lock);
		}
	}
	log_info("thread " << tid << " exits");
}

static void spawn()
{
	std::thread worker(threadWorker);
	worker.detach();
}

static void wait()
{
	while (sem.load()) {
		std::this_thread::yield();
	}
}

static void mpInit()
{
	descs.resize(nproc);
	for (unsigned i = 0; i < nproc; ++i)
		spawn();
}

static void mpDeinit()
{
	stopped = true;
	wakeCv.notify_all();
}


static void enqueue(JobDescriptor *job)
{
	while (!jobs.push_back(job))
		log_warn("worker queue full!");
	++sem;
	wakeCv.notify_one();
}

#define parallel_call(f) do { 									\
	unsigned delta = particles.size() / nproc; 					\
	for (unsigned i = 0; i < nproc; ++i) { 						\
		auto &desc = descs[i];									\
		desc.id = nproc - i - 1;								\
		desc.start = desc.id * delta;							\
		desc.end = i ? desc.start + delta : particles.size(); 	\
		desc.func = f##Worker;									\
		enqueue(&desc);											\
	} 															\
	wait();														\
} while(0)

void InitSPH(void)
{
	particles.reserve(MAX_PARTICLES);
	log_info("initializing dam break with " << DAM_PARTICLES << " particles");
	for (float y = EPS; y < VIEW_HEIGHT - EPS * 2.f; y += H)
	{
		for (float x = VIEW_WIDTH / 4; x <= VIEW_WIDTH / 2; x += H)
		{
			if (particles.size() < DAM_PARTICLES)
			{
				float jitter = static_cast<float>(arc4random()) / static_cast<float>(RAND_MAX);
				particles.push_back(Particle(x + jitter, y));
			}
			else
			{
				return;
			}
		}
	}
}

void IntegrateWorker(const JobDescriptor *job)
{
	auto start 	= particles.begin() + job->start;
	auto end 	= particles.begin() + job->end;

	for (auto it = start; it != end; ++it)
	{
		auto &p = *it;
		// forward Euler integration
		p.v += DT * p.f / p.rho;
		p.x += DT * p.v;

		// enforce boundary conditions
		if (p.x(0) - EPS < 0.f)
		{
			p.v(0) *= BOUND_DAMPING;
			p.x(0) = EPS;
		}
		if (p.x(0) + EPS > VIEW_WIDTH)
		{
			p.v(0) *= BOUND_DAMPING;
			p.x(0) = VIEW_WIDTH - EPS;
		}
		if (p.x(1) - EPS < 0.f)
		{
			p.v(1) *= BOUND_DAMPING;
			p.x(1) = EPS;
		}
		if (p.x(1) + EPS > VIEW_HEIGHT)
		{
			p.v(1) *= BOUND_DAMPING;
			p.x(1) = VIEW_HEIGHT - EPS;
		}
	}
}

void ComputeDensityPressureWorker(const JobDescriptor *job)
{
	auto start 	= particles.begin() + job->start;
	auto end 	= particles.begin() + job->end;

	Particle tmp(0.f, 0.f);
	for (auto it = start; it != end; ++it)
	{
		auto &pi = *it;
		pi.rho = 0.f;

		tmp.x = pi.x - Vector2f(H, 0);
		auto lower = std::lower_bound(particles.begin(), particles.end(), tmp);
		tmp.x = pi.x + Vector2f(H, 0);
		auto upper = std::upper_bound(particles.begin(), particles.end(), tmp);

		for (auto jit = lower; jit != upper; ++jit)
		{
			auto &pj = *jit;
			Vector2f rij = pj.x - pi.x;
			float r2 = rij.squaredNorm();

			if (r2 < HSQ)
			{
				// this computation is symmetric
				pi.rho += MASS * POLY6 * pow(HSQ - r2, 3.f);
			}
		}
		pi.p = GAS_CONST * (pi.rho - REST_DENS);
	}
}

void ComputeForcesWorker(const JobDescriptor *job)
{
	auto start 	= particles.begin() + job->start;
	auto end 	= particles.begin() + job->end;

	Particle tmp(0.f, 0.f);
	for (auto it = start; it != end; ++it)
	{
		auto &pi = *it;
		Vector2f fpress(0.f, 0.f);
		Vector2f fvisc(0.f, 0.f);

		tmp.x = pi.x - Vector2f(H, 0);
		auto lower = std::lower_bound(particles.begin(), particles.end(), tmp);
		tmp.x = pi.x + Vector2f(H, 0);
		auto upper = std::upper_bound(particles.begin(), particles.end(), tmp);

		for (auto jit = lower; jit != upper; ++jit)
		{
			auto &pj = *jit;
			if (&pi == &pj)
			{
				continue;
			}

			Vector2f rij = pj.x - pi.x;
			float r = rij.norm();

			if (r < H)
			{
				// compute pressure force contribution
				fpress += -rij.normalized() * MASS * (pi.p + pj.p) / (2.f * pj.rho) * SPIKY_GRAD * pow(H - r, 3.f);
				// compute viscosity force contribution
				fvisc += VISC * MASS * (pj.v - pi.v) / pj.rho * VISC_LAP * (H - r);
			}
		}
		Vector2f fgrav = G * MASS / pi.rho;
		pi.f = fpress + fvisc + fgrav;
	}
}

void sortParticles(void)
{
	std::sort(particles.begin(), particles.end());
}

static std::map<std::string, std::size_t> records;
#define record_time(content) do { \
	std::chrono::steady_clock::time_point begin;\
	if (records.count(#content) == 0) \
		records[#content] = 0;\
	begin = std::chrono::steady_clock::now();\
	content; \
	records[#content] += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin).count(); \
} while(0)

static void report_time()
{
	for (auto && item : records)
		log_info("time['" << item.first << "']: " << item.second / 1000ULL << "ms");
	std::cout.flush();
}

void Update(void)
{
	record_time(sortParticles());
	record_time(parallel_call(ComputeDensityPressure));
	record_time(parallel_call(ComputeForces));
	record_time(parallel_call(Integrate));

	glutPostRedisplay();
}

void InitGL(void)
{
	glClearColor(0.9f, 0.9f, 0.9f, 1);
	glEnable(GL_POINT_SMOOTH);
	glPointSize(H / 8.f);
	glMatrixMode(GL_PROJECTION);
}

void Render(void)
{
	static unsigned frame;
	static int t0, t1;

	glClear(GL_COLOR_BUFFER_BIT);

	glLoadIdentity();
	glOrtho(0, VIEW_WIDTH, 0, VIEW_HEIGHT, 0, 1);

	glColor4f(0.2f, 0.6f, 1.f, 1);
	glBegin(GL_POINTS);
	for (auto &p : particles)
	{
		glVertex2f(p.x(0), p.x(1));
	}
	glEnd();

	glutSwapBuffers();

	frame++;
	t1 = glutGet(GLUT_ELAPSED_TIME);

	if (t1 - t0 > 1000) {
		log_info_rewind(frame * 1000U / (t1 - t0) << " fps" << '\r');
	 	t0 = t1;
		frame = 0;
	}
}

void Keyboard(unsigned char c, __attribute__((unused)) int x, __attribute__((unused)) int y)
{
	switch (c)
	{
	case ' ':
		if (particles.size() >= MAX_PARTICLES)
		{
			log_warn("maximum number of particles reached");
		}
		else
		{
			unsigned int placed = 0;
			for (float y = VIEW_HEIGHT / 1.5f - VIEW_HEIGHT / 5.f; y < VIEW_HEIGHT / 1.5f + VIEW_HEIGHT / 5.f; y += H * 0.95f)
			{
				for (float x = VIEW_WIDTH / 2.f - VIEW_HEIGHT / 5.f; x <= VIEW_WIDTH / 2.f + VIEW_HEIGHT / 5.f; x += H * 0.95f)
				{
					if (placed++ < BLOCK_PARTICLES && particles.size() < MAX_PARTICLES)
					{
						particles.push_back(Particle(x, y));
					}
				}
			}
		}
		break;
	case 'r':
	case 'R':
		particles.clear();
		InitSPH();
		break;
	}
}

int main(int argc, char **argv)
{
	mpInit();

	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInit(&argc, argv);
	glutCreateWindow("Müller SPH");
	glutDisplayFunc(Render);
	glutIdleFunc(Update);
	glutKeyboardFunc(Keyboard);
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);

	InitGL();
	InitSPH();

	glutMainLoop();

	mpDeinit();
	report_time();
	return 0;
}
