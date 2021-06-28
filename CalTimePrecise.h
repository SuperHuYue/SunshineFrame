#pragma once
#include <windows.h>
/*
*精确计算时间的类（us），两者一定 要配合使用
*/
class CalTimePrecise
{
public:
	CalTimePrecise() {};
	~CalTimePrecise() {};
	void SetStartTime() {
		QueryPerformanceFrequency(&litmp);
		// 获得计数器的时钟频率
		dfFreq = (double)litmp.QuadPart;
		QueryPerformanceCounter(&litmp);
		// 获得初始值
		QPart1 = litmp.QuadPart;
	};
	double Elapse()
	{
		QueryPerformanceCounter(&litmp);
		// 获得终止值
		QPart2 = litmp.QuadPart;
		dfMinus = (double)(QPart2 - QPart1);
		dfTime = dfMinus / dfFreq;
		return dfTime;
	}

private:
	LARGE_INTEGER litmp;
	LONGLONG QPart1, QPart2;
	double dfMinus, dfFreq, dfTime;
};


