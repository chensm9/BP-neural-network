#ifndef __UTILS_H__
#define __UTILS_H__

#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace std;

#define PROGRESS_DISP_CHAR '*'

// 随机整数数[x, y]
inline int RandInt(int x, int y)
{ 
	return rand() % (y - x + 1) + x; 
}

// 随机浮点数（0， 1）
inline double RandFloat()
{ 
	return (rand()) / (RAND_MAX + 1.0); 
}

// 随机布尔值
inline bool RandBool()
{
	return RandInt(0, 1) ? true : false;
}

// 随机浮点数（-1， 1）
inline double RandomClamped()
{ 
	return rand() % 1000 * 0.001 - 0.5;
}


// 高斯分布
inline double RandGauss()
{
	static int	  iset = 0;
	static double gset = 0;
	double fac = 0, rsq = 0, v1 = 0, v2 = 0;

	if (iset == 0)
	{
		do
		{
			v1 = 2.0*RandFloat() - 1.0;
			v2 = 2.0*RandFloat() - 1.0;
			rsq = v1*v1 + v2*v2;
		} while (rsq >= 1.0 || rsq == 0.0);

		fac = sqrt(-2.0*log(rsq) / rsq);
		gset = v1*fac;
		iset = 1;
		return v2*fac;
	}
	else
	{
		iset = 0;
		return gset;
	}
}


inline uint64_t timeNowMs()
{
    std::chrono::time_point<std::chrono::system_clock> p2 =
        std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>
        (p2.time_since_epoch()).count();
}

#endif // !__NEURON_UTILS_H__