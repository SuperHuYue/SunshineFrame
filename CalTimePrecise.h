#pragma once
#include <windows.h>
/*
*��ȷ����ʱ����ࣨus��������һ�� Ҫ���ʹ��
*/
class CalTimePrecise
{
public:
	CalTimePrecise() {};
	~CalTimePrecise() {};
	void SetStartTime() {
		QueryPerformanceFrequency(&litmp);
		// ��ü�������ʱ��Ƶ��
		dfFreq = (double)litmp.QuadPart;
		QueryPerformanceCounter(&litmp);
		// ��ó�ʼֵ
		QPart1 = litmp.QuadPart;
	};
	double Elapse()
	{
		QueryPerformanceCounter(&litmp);
		// �����ֵֹ
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


