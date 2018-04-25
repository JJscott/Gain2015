#pragma once

// opencv
#include <opencv2/core.hpp>

// std
#include <random>
#include <type_traits>

inline cv::Point clampToRect(const cv::Point& p, const cv::Rect& rect) {
	using namespace cv;
	using namespace std;
	return Point(max(rect.x, min(p.x, rect.x + rect.width)), max(rect.y, min(p.y, rect.y + rect.height)));
}

inline cv::Point clampToMat(const cv::Point& p, const cv::Mat& mat) {
	using namespace cv;
	using namespace std;
	return Point(max(0, min(p.x, mat.cols-1)), max(0, min(p.y, mat.rows-1)));
}


// sample takes on int coordinates (like opencv)
template<typename T>
T sample(cv::Mat m, cv::Vec2f p) {
	using namespace cv;
	using namespace std;
	p += Vec2f(.5);
	T r = m.at<T>(0, 0) - m.at<T>(0, 0); // lol, get a proper zero value by cheating
	for (int j = 0; j < 2; j++) {
		for (int i = 0; i < 2; i++) {
			Point p1(floor(j + p[0] - 0.5f), floor(i + p[1] - 0.5f));
			Vec2f d(1.f - abs(p1.x + 0.5f - p[0]), 1.f - abs(p1.y + 0.5f - p[1]));
			Point cp = clampToMat(p1, m);
			r += m.at<T>(cp) * d[0] * d[1];
		}
	}
	return r;
}

namespace util {

	namespace detail {
		inline void to_stream(std::ostream &) {}

		template <typename ArgT0, typename ...ArgTs>
		inline void to_stream(std::ostream &out, const ArgT0 &arg, const ArgTs &... args) {
			out << arg;
			to_stream(out, args...);
		}

		template <typename T, typename = void>
		struct distribution {};

		template <typename T>
		using distribution_t = typename distribution<T>::type;

		template <typename T>
		struct distribution<T, std::enable_if_t<std::is_integral<T>::value>> {
			using type = std::uniform_int_distribution<
				std::conditional_t<
					sizeof(T) < sizeof(short),
					std::conditional_t<
						std::is_signed<T>::value,
						short,
						unsigned short
					>,
					T
				>
			>;
		};

		template <typename T>
		struct distribution<T, std::enable_if_t<std::is_floating_point<T>::value>> {
			using type = std::uniform_real_distribution<T>;
		};

		// singleton for random engine
		inline auto & random_engine() {
			static thread_local std::default_random_engine re{ std::random_device()() };
			return re;
		}
	}

	// helper function to return a string from given arguments
	// requires every argument has an overload for the '<<' operator
	template <typename ...ArgTs>
	inline std::string stringf(const ArgTs &... args) {
		std::ostringstream oss;
		detail::to_stream(oss, args...);
		return oss.str();
	}


	inline void reset_random(long seed = 0) {
		detail::random_engine() = std::default_random_engine(seed);
	}

	// return a random value of T in range [lower, upper)
	template <typename T, typename P>
	inline T random(P lower, P upper) {
		using dist_t = detail::distribution_t<T>;
		dist_t dist(typename dist_t::param_type(lower, upper));
		return dist(detail::random_engine());
	}

	// return a random value of T in range [0, upper)
	template <typename T, typename P>
	inline T random(P upper) {
		return random<T, P>(std::decay_t<P>(0), upper);
	}

	// return a random value of T in the range defined by associated uniform distribution
	template <typename T>
	inline T random() {
		using dist_t = detail::distribution_t<T>;
		dist_t dist;
		return dist(detail::random_engine());
	}
}