#pragma once
#include "CMatrix.hpp"
#include<iostream>
#include <functional>
#include <assert.h>
#include <iomanip>
#include "CalTimePrecise.h"
namespace SunshineFrame {
	namespace Algebra {
		using namespace std;
		double const pi = 4 * std::atan(1.0);
		CMatrix::CMatrix(std::list<int> shape):
			 m_ptrData(nullptr)
		{
			m_ndim = shape.size();
			int ncount = 0;
			m_nTotalSize = 1;
			bool bNegaApp = false;
			for (auto i = shape.begin(); i != shape.end(); i++)
			{
				ncount++;
				if (*i < 0)bNegaApp = true;
				m_nTotalSize *= (*i);
			}
			if (bNegaApp == false)//shape有负项不会为其搭载内存，同时map也不会有项目
			{
				//m_ptrData = std::shared_ptr<MatrixDataType[]>(new MatrixDataType[m_nTotalSize]);
				//m_ptrData = std::shared_ptr<MatrixDataType[]>(new MatrixDataType[m_nTotalSize]);
				m_ptrData = std::shared_ptr<MatrixDataType[]>(new MatrixDataType[m_nTotalSize]);
				//memset(m_ptrData.get(), 0, sizeof(MatrixDataType) * m_nTotalSize);
				if (nullptr == m_ptrData) { cerr << "Not enough memory" << endl; return; };
			}
			m_listShape = shape;
			CalAxisCarry();
		}

		CMatrix CMatrix::zeros(list<int> shape)
		{
			CMatrix tep_Cache(shape);
			
			memset(tep_Cache.m_ptrData.get(), 0, sizeof(MatrixDataType) * tep_Cache.m_nTotalSize);
			return tep_Cache;
		}

		void CMatrix::CalAxisCarry()
		{
			if (m_nTotalSize < 0)
			{
				m_mapAxisCarryOver.clear();
				cout << "Size尚不确定，跳过CalAxisCarry部分,同时尺寸清零..." << endl;
				return;
			}
			int ndim = m_listShape.size();
			auto it_shape = m_listShape.rbegin();
			int total_size = 1;
			for (int i = 0; i < ndim; ++i)
			{
				assert(it_shape != m_listShape.rend());
				m_mapAxisCarryOver[i] = total_size;
				total_size *= *it_shape;
				it_shape++;
			}

		}

		void CMatrix::zeros()
		{
			memset(m_ptrData.get(), 0, sizeof(MatrixDataType) * m_nTotalSize);
		}

		void CMatrix::ones()
		{
			for (int i = 0; i < m_nTotalSize; ++i)
			{
				m_ptrData[i] = 1;
			}
		}

		void CMatrix::fixNumbers(const MatrixDataType& num)
		{
			for (int i = 0; i < m_nTotalSize; ++i)
			{
				m_ptrData[i] = num;
			}
		}

		void CMatrix::random_normalize(const double& mu, const double& sigma, int seed)
		{
			int total_size = gettotalsize();
			auto ptr_data = getdataptr();
			int seed_index = seed;
			for (int i = 0; i < total_size; i++)
			{
				if(seed != -1)std::srand(seed_index);		
				seed_index++;
				double r1 = (std::rand() + 1.0) / (RAND_MAX + 1.0); // gives equal distribution in (0, 1]
				double r2 = (std::rand() + 1.0) / (RAND_MAX + 1.0);
				ptr_data[i] = mu + sigma * std::sqrt(-2 * std::log(1 - r1))*std::cos(2 * pi*r2);
			}
			
			//mu + sigma * std::sqrt(-2 * std::log(r1))*std::cos(2 * pi*r2);
		}


		CMatrix CMatrix::T()const
		{
			if (m_listShape.size() != 2) { cerr << "Fail: Not Support Dim bigger than two shape..." << endl; return *this; }
			CMatrix output = change_axis(*this, 1, 0);
			output.CalAxisCarry();
			return output;
			//const int& a = m_listShape.front();
			//const int& b = m_listShape.back();
			//assert(m_ptrData != nullptr);
			//auto pdata = m_ptrData;
			//double* tep_cache = new double[m_nTotalSize];
			//memset(tep_cache, 0, sizeof(tep_cache) * m_nTotalSize);
			////CMatrix tep = *this;
			////tep.print();
			//for (int i = 0; i < a; ++i)
			//{
			//	for (int j = 0; j < b; ++j)
			//	{
			//		assert(i * b + j < m_nTotalSize);
			//		double www = pdata[i * b + j];
			//		assert(j * a + i < m_nTotalSize);			
			//		tep_cache[j * a + i] = www;
			//	}
			//}
			//CMatrix output(list<int>{b, a});
			//memcpy(output.m_ptrData.get(), tep_cache, sizeof(double) * m_nTotalSize);
			//delete[] tep_cache;
			//tep_cache = NULL;
			//return output;
		}
		/*
		The Broadcasting Rule
		In order to broadcast, the size of the trailing axes for both arrays in an 
		operation must either be the same size or one of them must be one.
		*/
		bool CMatrix::broadcastRule(const CMatrix& lhs, const CMatrix& rhs,list<int>& out_shape)
		{
			out_shape.clear();
			auto it_lhs = lhs.m_listShape.rbegin();
			auto it_rhs = rhs.m_listShape.rbegin();
			do
			{
				if (it_lhs != lhs.m_listShape.rend() && it_rhs != rhs.m_listShape.rend())
				{
					if (*it_lhs > *it_rhs && 1 == *it_rhs) { out_shape.push_front(*it_lhs); it_lhs++; it_rhs++; continue; }
					if (*it_lhs < *it_rhs && 1 == *it_lhs) { out_shape.push_front(*it_rhs); it_lhs++; it_rhs++; continue; }
					if (*it_lhs == *it_rhs) { out_shape.push_front(*it_lhs); it_lhs++; it_rhs++; continue; };
					return false;
				}
				if (it_lhs == lhs.m_listShape.rend())
				{
					for (;it_rhs != rhs.m_listShape.rend(); it_rhs++)
					{
						out_shape.push_front(*it_rhs);
					}
					return true;
				}
				if (it_rhs == rhs.m_listShape.rend())
				{
					for (; it_lhs != lhs.m_listShape.rend(); it_lhs++)
					{
						out_shape.push_front(*it_lhs);
					}
					return true;
				}
				it_rhs++;
				it_lhs++;
			} while (true);
		}


		CMatrix CMatrix::genMatByBroadcastRule(const CMatrix& in, const std::list<int>& shape) {
			//检查size
			int finalDim = shape.size();
			int nowDim = in.m_listShape.size();
			if (finalDim < nowDim)throw std::runtime_error("Err:genMatByBroadcastRule shut down, check broadcastRule, function have bugs... ");
			auto expandFunc = [](const CMatrix& in, const std::list<int>& shape)->SunshineFrame::Algebra::CMatrix {
				//此函数为同维度下将shape为1的部分
				std::function<void(const CMatrix&, const std::list<int>&)>f;
				CMatrix finalMat;
				f = [&](const CMatrix& it, const std::list<int>& shape)->void{
					auto tmpShape = it.shape();
					if (tmpShape.size() != shape.size())throw std::runtime_error("Err:expandFunc shut down bugs in genMayByBroacastRule, please check...");
					auto itr_it = tmpShape.rbegin();
					int higestAxis = shape.size() - 1;
					int nowAxis = 0;
					CMatrix sinChangeOut;
					bool finish = true;
					for (auto itr_final = shape.rbegin(); itr_final != shape.rend(); ++itr_final,++itr_it, ++nowAxis) {
						if (*itr_it != *itr_final && *itr_it == 1) {
							auto needCopyObj = change_axis(it, nowAxis, higestAxis);
							auto tmp = needCopyObj.shape();
							if (tmp.front() != 1)throw std::runtime_error("Err:expandFunc shut down bugs in change_axis, please check...");
							tmp.pop_front();
							tmp.push_front(*itr_final);
							CMatrix target(tmp);
							for (int i = 0; i < *itr_final; ++i) {
								memcpy(target.getdataptr().get() + i * needCopyObj.gettotalsize(), needCopyObj.getdataptr().get(), needCopyObj.gettotalsize() * sizeof(MatrixDataType));
							}
							target = change_axis(target, higestAxis, nowAxis);
							sinChangeOut = target;
							finish = false;
							break;
						}
					}
					if (finish) {
						finalMat = it;
						return;
					}
					f(sinChangeOut, shape);
					return;
				};

				f(in, shape);
				return finalMat;
			};
			if (finalDim != nowDim) 
			{
				//拷贝成维度相等内容
				int needCopySize = in.gettotalsize();
				int needCopyTimes = 1;
				std::list<int>tmp_shape = in.m_listShape;
				int count = 1;
				for (auto itr = shape.rbegin(); itr != shape.rend(); ++itr,++count) {
					if (count <= nowDim)continue;
					tmp_shape.push_front(*itr);
					needCopyTimes *= *itr;
				}
				CMatrix tmp(tmp_shape);
				for (int i = 0; i < needCopyTimes; ++i) {
					memcpy(tmp.getdataptr().get() + (size_t)i * needCopySize, in.getdataptr().get(), needCopySize * sizeof(MatrixDataType));
				}
				return expandFunc(tmp, shape);
			}
			else 
			{
				return expandFunc(in, shape);

			}
		}
		CMatrix CMatrix::ones(list<int> shape)
		{
			CMatrix tep_Cache(shape);
			for (int i = 0; i < tep_Cache.m_nTotalSize; ++i)
			{
				tep_Cache.m_ptrData[i] = 1.0;
			}
			return tep_Cache;
		}


		CMatrix::CMatrix(const CMatrix & cpy)
			:m_ptrData(nullptr)
		{
			this->m_listShape = cpy.m_listShape;
			this->m_mapAxisCarryOver = cpy.m_mapAxisCarryOver;
			this->m_nTotalSize = cpy.m_nTotalSize;
			this->m_ndim = cpy.m_ndim;

			//this->m_ptrData = std::make_shared<double>(new double[m_nTotalSize], std::default_delete<double[]>());
			this->m_ptrData = std::shared_ptr<MatrixDataType[]>(new MatrixDataType[m_nTotalSize]);

			memcpy(this->m_ptrData.get(), cpy.m_ptrData.get(), sizeof(MatrixDataType) * this->m_nTotalSize);
				
		}

		CMatrix& CMatrix::operator=(CMatrix &&rhs) noexcept
		{
			if (this != &rhs) {
				m_ptrData = rhs.m_ptrData;
				m_listShape = rhs.m_listShape;
				m_mapAxisCarryOver = rhs.m_mapAxisCarryOver;
				m_nTotalSize = rhs.m_nTotalSize;
				m_ndim = rhs.m_ndim;
				rhs.m_ptrData = nullptr;

			}
			return *this;
		}
		void CMatrix::matrixFeed(std::list<MatrixDataType> data)
		{
			if (data.size() > gettotalsize()) {
				throw std::runtime_error("Can not feed a valid data..data not fit");//数据与容器大小不匹配
			}
			int idx = 0;
			for (auto i : data) {
				m_ptrData[idx++] = i;
			}

		}

		CMatrix::CMatrix(CMatrix &&cpy) noexcept
			:m_ptrData(cpy.m_ptrData),m_listShape(cpy.m_listShape),m_mapAxisCarryOver(cpy.m_mapAxisCarryOver),
			m_nTotalSize(cpy.m_nTotalSize),m_ndim(cpy.m_ndim)
		{
			cpy.m_ptrData = nullptr;
		}

		CMatrix& CMatrix::operator=(const CMatrix & rhs)
		{
			if (&rhs == this)return *this;
			if (m_ptrData != nullptr) {  m_ptrData = nullptr; }
			this->m_listShape = rhs.m_listShape;
			this->m_mapAxisCarryOver = rhs.m_mapAxisCarryOver;
			this->m_nTotalSize = rhs.m_nTotalSize;
			this->m_ndim = rhs.m_ndim;

			//this->m_ptrData = std::make_shared<double>(new double[m_nTotalSize], std::default_delete<double[]>());
			this->m_ptrData = std::shared_ptr<MatrixDataType[]>(new MatrixDataType[m_nTotalSize]);

			memcpy(this->m_ptrData.get(), rhs.m_ptrData.get(), sizeof(MatrixDataType) * this->m_nTotalSize);
				
			return *this;
		}
		/*
		*获得pos索引里面指定位置的数据的指针(脱离share_ptr管控，他是危险的)
		*/
		MatrixDataType* CMatrix::getPosDataPtr(list<int>pos)
		{
			if (!cekIdxOk(pos))
			{
				cerr << "Get data Fail...." << endl;
				return 0;
			}
			int index = CMatrix::matSizefrompos(*this, pos);
			MatrixDataType* out_ptr = getdataptr().get() + index;
			return out_ptr;
		}

		MatrixDataType CMatrix::getPosData(list<int>pos)
		{
			if (!cekIdxOk(pos))
			{
				cerr << "Get data Fail...." << endl;
				return 0;
			}
			int index = CMatrix::matSizefrompos(*this, pos);
			MatrixDataType out_ptr = getdataptr()[index];
			return out_ptr;
		}
		CMatrix CMatrix::operator+(const int& rhs)
		{
			
			CMatrix tep_cache(*this);
			for (int i = 0; i < tep_cache.gettotalsize(); i++)
			{
				tep_cache.getdataptr()[i] += rhs;
			}
			return tep_cache;


		}

		CMatrix CMatrix::operator+(const MatrixDataType& rhs)
		{
			
			CMatrix tep_cache(*this);
			for (int i = 0; i < tep_cache.gettotalsize(); i++)
			{
			tep_cache.getdataptr()[i] += rhs;
			}
			return tep_cache;


		}



		/*
		* 矩阵求和,位加法运算，允许broadcast
		*/
		CMatrix CMatrix::operator+(const CMatrix & rhs)
		{	
			std::list<int> out_shape;
			bool ok = CMatrix::broadcastRule(*this, rhs, out_shape);
			CMatrix tmpThis = *this;
			CMatrix tmpRhs = rhs;
			if (ok) {
				tmpThis = CMatrix::genMatByBroadcastRule(tmpThis, out_shape);
				tmpRhs = CMatrix::genMatByBroadcastRule(tmpRhs, out_shape);
				CMatrix out(tmpThis.shape());
				for (int i = 0; i < out.m_nTotalSize; ++i) {
					*(out.getdataptr().get() + i) = *(tmpThis.getdataptr().get() + i) + *(tmpRhs.getdataptr().get() + i);
				}
				return out;
			}
			throw std::runtime_error("Err: 	CMatrix CMatrix::operator+(const CMatrix & rhs)");
		}
		CMatrix CMatrix::operator-(const CMatrix& rhs) {
			auto a = -1 * rhs;
			return *this + a;
		}
		


		CMatrix& CMatrix::operator+=(const CMatrix& rhs) 
		{
			return *this = *this + rhs;//调用加法法则
		}

		CMatrix& CMatrix::operator-=(const CMatrix& rhs) 
		{
			return *this = *this - rhs;
		}

		CMatrix& CMatrix::operator*=(const CMatrix& rhs) 
		{
			return *this = *this * rhs;
		}
		bool CMatrix::operator==(const CMatrix& rhs) const{
			std::list<int> out_shape;
			bool ok = CMatrix::broadcastRule(*this, rhs, out_shape);
			CMatrix tmpThis = *this;
			CMatrix tmpRhs = rhs;
			if (ok) {
				tmpThis = CMatrix::genMatByBroadcastRule(tmpThis, out_shape);
				tmpRhs = CMatrix::genMatByBroadcastRule(tmpRhs, out_shape);
				for (int i = 0; i < tmpThis.m_nTotalSize; ++i) {
					if (tmpThis.getdataptr()[i] != tmpRhs.getdataptr()[i])return false;
				}
				return true;
			}
			return false;
		}



		/*
		*位乘法运算
		*/
		CMatrix CMatrix::operator*(const CMatrix & rhs)const
		{
			std::list<int> out_shape;
			bool ok = CMatrix::broadcastRule(*this, rhs, out_shape);
			CMatrix tmpThis = *this;
			CMatrix tmpRhs = rhs;
			if (ok) {
				tmpThis = CMatrix::genMatByBroadcastRule(tmpThis, out_shape);
				tmpRhs = CMatrix::genMatByBroadcastRule(tmpRhs, out_shape);
				CMatrix out(tmpThis.shape());
				for (int i = 0; i < out.m_nTotalSize; ++i) {
					*(out.getdataptr().get() + i) = *(tmpThis.getdataptr().get() + i) * *(tmpRhs.getdataptr().get() + i);
				}
				return out;
			}
			throw std::runtime_error("Err:can not do:operator*(const CMatrix & rhs)");
		}

		CMatrix CMatrix::operator*(const MatrixDataType& rhs)
		{
			if(this->m_ptrData == nullptr)throw std::runtime_error("Err:CMatrix CMatrix::operator*(const MatrixDataType& rhs)");
			CMatrix tep_cache = *this;
			for (int i = 0; i < m_nTotalSize; i++)
			{
				tep_cache.m_ptrData[i] *= rhs;
			}
			return tep_cache;
		}

		CMatrix CMatrix::operator*(const int& rhs)
		{
			
			if(this->m_ptrData == nullptr)throw std::runtime_error("Err:CMatrix CMatrix::operator*(const int& rhs)");
			CMatrix tep_cache = *this;
			for (int i = 0; i < m_nTotalSize; i++)
			{
				tep_cache.m_ptrData[i] *= rhs;
			}
			return tep_cache;
		}

		CMatrix operator*(const int& lhs, const CMatrix & rhs)
		{
			CMatrix tep_cache = rhs;
			for (int i = 0; i < tep_cache.gettotalsize(); i++)
			{
				tep_cache.getdataptr()[i] *= lhs;
			}
			return tep_cache;
		}

		CMatrix operator*(const MatrixDataType& lhs, const CMatrix & rhs)
		{
			CMatrix tep_cache = rhs;
			for (int i = 0; i < tep_cache.gettotalsize(); i++)
			{
				tep_cache.getdataptr()[i] *= lhs;
			}
			return tep_cache;
		}

		CMatrix operator+(const MatrixDataType& lhs, const CMatrix & rhs)
		{
			CMatrix tep_cache(rhs);
			for (int i = 0; i < tep_cache.gettotalsize(); i++)
			{
				tep_cache.getdataptr()[i] += lhs;
			}
			return tep_cache;
		}

		/*
		*高维向量乘法运算：高于二维的维度，挨个遍历，便可得到一个二维矩阵，对这两个二维矩阵进行线性代数里的矩阵乘法
		*因此，对于高于二维的维度每一个都必须相同，遍历出来的二维矩阵需要满足向量乘法法则
		*/
		CMatrix CMatrix::matmul(const CMatrix& lhs, const CMatrix& rhs)
		{
			//高维支持
			if (lhs.m_ndim != rhs.m_ndim || lhs.m_ndim <= 1) {
				throw std::runtime_error("Err: matmul err reason dim not equal or less 2");
			}
			//检测是否满足相乘的条件
			int count = lhs.m_listShape.size() - 1;
			auto lhs_itr = lhs.m_listShape.begin();
			auto rhs_itr = rhs.m_listShape.begin();
			for (int i = count; i >= 2; --i) {
				if (*lhs_itr != *rhs_itr) {
					throw std::runtime_error("Err::matmul dim bigger than 3 not equal");
				}
				lhs_itr = std::next(lhs_itr);
				rhs_itr = std::next(rhs_itr);
			}
			//last two dim check
			lhs_itr = std::next(lhs_itr);
			if (*lhs_itr != *rhs_itr) {
				throw std::runtime_error("Err::matmul dim last 2 dim not fit");
			}
			//
			int tep_count = 1;
			int times = 1;//开辟内存倍数
			int single_mul_size_out = 1;//单个矩阵乘法输出需要偏移的大小
			int single_mul_size_rhs = 1;//单个矩阵乘法rhs需要偏移的大小
			int single_mul_size_lhs = 1;//单个矩阵乘法lhs需要偏移的大小
			list<int> out_shape;//eg:[6;7;4] * [6;4;5]====>[6;7;5]
			auto it_rhs = rhs.m_listShape.rbegin();
			int two_dim_row = 1;
			int two_dim_col = 1;
			for (auto it_lhs = lhs.m_listShape.rbegin(); it_lhs != lhs.m_listShape.rend(); ++it_lhs,++it_rhs,++tep_count)
			{
				if (tep_count == 1) { 
					out_shape.push_front(*it_rhs);
					two_dim_col = *it_rhs;
					single_mul_size_out *= *it_rhs;
					single_mul_size_lhs *= *it_lhs;
					single_mul_size_rhs *= *it_rhs;
				}
				else if (tep_count == 2) {
					out_shape.push_front(*it_lhs);
					two_dim_row = *it_lhs;
					single_mul_size_out *= *it_lhs;
					single_mul_size_lhs *= *it_lhs;
					single_mul_size_rhs *= *it_rhs;
				}
				else {
					out_shape.push_front(*it_rhs);
				}
			}
			times = lhs.m_nTotalSize / single_mul_size_lhs;

			CMatrix outMatrix(out_shape);
			LONGLONG outtotalsize = outMatrix.gettotalsize();
			LONGLONG lhstotalsize = lhs.gettotalsize();
			LONGLONG rhstotalsize = rhs.gettotalsize();
			auto ptr_out = outMatrix.getdataptr();
			auto ptr_lhs = lhs.getdataptr();
			auto ptr_rhs = rhs.getdataptr();

			int feed_col = lhs.m_listShape.back();		
			for (int i = 0; i < times; i++)
			{
				for (int index_row = 0; index_row < two_dim_row; ++index_row)
				{
					for (int index_col = 0; index_col < two_dim_col; index_col++)
					{
						//进行计算
						MatrixDataType tep_total = 0;
						assert(i * (size_t)single_mul_size_out + (size_t)index_row * outMatrix.m_listShape.back() + index_col < outtotalsize);//out越界控制
						for (int j = 0; j < lhs.m_listShape.back(); ++j)
						{
							
							assert(i * (size_t)single_mul_size_lhs + (size_t)index_row * lhs.m_listShape.back() + j < lhstotalsize);//lhs越界控制
							assert(i * (size_t)single_mul_size_rhs + (size_t)j * rhs.m_listShape.back() + index_col < rhstotalsize);//rhs越界控制
							tep_total +=
								ptr_lhs[i * single_mul_size_lhs + (size_t)index_row * lhs.m_listShape.back() + j] *
								ptr_rhs[i * single_mul_size_rhs + (size_t)j * rhs.m_listShape.back() + index_col];
						}
						ptr_out[i * single_mul_size_out + (size_t)index_row * outMatrix.m_listShape.back() + index_col] = tep_total;
					}

				}
			}
			return outMatrix;
		}

		/*
		* 检测索引值是否合理
		*/
		bool CMatrix::cekIdxOk(list<int> pos)
		{
			if (pos.size() != m_listShape.size())
			{
				cerr << "Not a reasonable size....." << endl;
				return false;
			}
			auto InnerBegin = m_listShape.begin();
			auto CheckBegin = pos.begin();
			for (size_t i = 0; i < pos.size(); ++i)
			{
				if ((*CheckBegin) < 0 || (*CheckBegin) >= (*InnerBegin))
				{
					cerr << "Not a reasonable index....." << endl;
					return false;
				}
			}
			return true;

		}
		/*
		* 打印矩阵数据信息
		* 主要思想：首先依据纬度给出 [ 的个数，同时，以shape=（3，4，5）为例，每当出现5，20，60的倍数的数，就会给一个]，ex：第20的数同时能被5和20整除
		*这里就会给两个]]，其他的依旧是如此，下一个的[则是以上一个]的个数去不足
		*/
		void CMatrix::print()const
		{
			if (m_ptrData == nullptr)return;
			int dim = this->m_ndim;
			list<int> times;
			//画出第一步的[以及计算倍数
			for (int i = 0; i < dim; ++i)
			{
				cout << "[";
				int tep_count = 1;
				auto it = m_listShape.rbegin();
				for (int j = 0; j < i + 1; ++j)
				{
					assert(it != m_listShape.rend());
					tep_count *= *it;
					it++;
					//tep_count *= 
				}
				times.push_back(tep_count);
			}
			/////////////////////////////////////////////////////////
			int count = 0;
			for (int i = 0; i < m_nTotalSize; ++i)
			{
				cout.setf(ios::fixed);
				cout << " " << setprecision(std::numeric_limits<Algebra::MatrixDataType>::max_digits10) <<m_ptrData[i];
				//cout << " "  << *((MatrixDataType*)m_ptrData + i);
				count = 0;
				for (auto it = times.begin(); it != times.end(); ++it)
				{

					if ((i + 1) % ((*it)) == 0) { cout << "]"; count++; }
					else break;
				}
				if (count == 0)cout << ",";
				else cout << "," << endl << endl;

				if (i == m_nTotalSize - 1)return;
				for (int i = 0; i < count; ++i)
				{
					cout << "[";
				}
			}
		}

		bool CMatrix::reshape(list<int> entershape)
		{
			//assert(entershape.size() != 0);需要允许reshape为标量
			int entersize = 1;
			bool bGoThrough = false;
			for (auto it = entershape.begin(); it != entershape.end(); ++it)
			{
				entersize *= *it;
				//如果存在-1则不进行合理性判定
				if (*it == -1) { bGoThrough = true; break; }
			}
			if (entersize != m_nTotalSize && bGoThrough == false) { cerr << "Can't be reshaped: Data not enough..." << endl; return false; };
			//just change the shape para is enough...
			easy_changeshape(entershape);
			CalAxisCarry();
			return true;
		}
		CMatrix CMatrix::mean(const CMatrix& enter, const int& axis, const bool& keepdim)
		{
			if (enter.m_ndim - 1 < axis)throw std::runtime_error("Err: CMatrix mean axis overflow..");
			CMatrix changeAxisMat = change_axis(enter, axis, 0);
			auto tmpShape = changeAxisMat.m_listShape;
			int loopTimes = 1;
			int offset = tmpShape.back();
			tmpShape.pop_back();
			for (auto i : tmpShape) {
				loopTimes *= i;
			}
			tmpShape.push_back(1);
			CMatrix out(tmpShape);
			for (int i = 0; i < loopTimes; ++i) {
				MatrixDataType mean = 0;
				for (int j = 0; j < offset; ++j) {
					int size = i * offset + j;
					mean += changeAxisMat.m_ptrData[size];
				}
				mean = mean / offset;
				out.m_ptrData[i] = mean;
			}
			out = change_axis(out, 0, axis);
			if(!keepdim)
			{
				std::list<int>out_shape;
				for (auto i : out.m_listShape) {
					if (i != 1) {
						out_shape.push_back(i);
					}
				}
				out.reshape(out_shape);
			}
			return out;
		}


		CMatrix CMatrix::change_axis(const CMatrix & enter, const int& from, const int& to)
		{
			//加速
			if (from == to) {
				return enter;
			}
			int max_axis_num = enter.m_ndim - 1;
			assert(max_axis_num >= from && max_axis_num >= to && from >= 0 && to >= 0);
			list<int> enter_cache = enter.m_listShape;
			{
				int tep_count = 0;
				int a_from = 0;
				int b_to = 0;
				auto a_idx = enter_cache.rbegin();
				auto b_idx = enter_cache.rbegin();
				for (auto it = enter_cache.rbegin(); it != enter_cache.rend(); ++it, ++tep_count)
				{
					if (tep_count == from) {
						a_from = *it;
						a_idx = it;
					}
					if (tep_count == to) {
						b_to = *it;
						b_idx = it;
					}
				}
				*a_idx = b_to;
				*b_idx = a_from;
			}
			CMatrix output(enter_cache);

			list<int> pos;
			for (int i = 0; i < enter.m_nTotalSize; ++i)
			{
				list<int>tep_from = matPosfromsize(enter, i);
				MatrixDataType ori_data = enter.m_ptrData[i];
				int axis_count = 0;
				int from_axis_data = 0;
				int to_axis_data = 0;
				auto it_from_idx = tep_from.rbegin();
				auto it_to_idx = tep_from.rbegin();
				for (auto it = tep_from.rbegin(); it != tep_from.rend(); ++it, ++axis_count)
				{
					if (axis_count == from) {
						from_axis_data = *it;
						it_from_idx = it;
					}
					if (axis_count == to) {
						to_axis_data = *it;
						it_to_idx = it;
					}
				}
				*it_from_idx = to_axis_data;
				*it_to_idx = from_axis_data;		
				MatrixDataType* tar_pos_data = output.getPosDataPtr(tep_from);
				*tar_pos_data = ori_data;
			}
			//output.reshape(enter_cache);
			return output;
		}

		void CMatrix::easy_changeshape(std::list<int> shape)
		{

			this->m_listShape.clear();
			m_listShape = shape;
			m_ndim = shape.size();
			CalAxisCarry();
		}
	
		int CMatrix::matSizefrompos(const CMatrix & enter, const std::vector<int> & pos)
		{	
			assert(enter.m_mapAxisCarryOver.size() == pos.size() && enter.m_ndim == pos.size());
			if (pos.size() == 0) { return 0; }
			int final = 0;
			int index = 0;
			auto test = enter.m_mapAxisCarryOver;
			for (auto test_pos = pos.rbegin(); test_pos != pos.rend(); ++test_pos)
			{
				assert(index >= 0);
				final += (test[index++] * (*test_pos));		
			}
			return std::move(final);
		}
		int CMatrix::matSizefrompos(const CMatrix & enter, const std::list<int> & pos)
		{	
			assert(enter.m_mapAxisCarryOver.size() == pos.size() && enter.m_ndim == pos.size());
			if (pos.size() == 0) { return 0; }
			int final = 0;
			int index = 0;
			auto test = enter.m_mapAxisCarryOver;
			//auto test_right = enter.m_listShape.begin();
			for (auto test_pos = pos.rbegin(); test_pos != pos.rend(); ++test_pos)
			{
				assert(index >= 0);
				final += (test[index++] * (*test_pos));		
			}
			return std::move(final);
		}

		


		std::list<int> CMatrix::matPosfromsize(const CMatrix & enter, const int& size)
		{
			std::list<int> pos;
			int tep_size = size;
			auto map = enter.m_mapAxisCarryOver;
			assert(map.size() == enter.m_ndim);
			for (int i = enter.m_ndim - 1; i >= 0; --i)
			{
				int tep_index = tep_size / map[i];
				pos.push_back(tep_index);
				if (tep_index != 0) {
					tep_size -= tep_index * map[i];
				}
			}
			return pos;
		}
		void CMatrix::setData(const std::list<int>& pos, const MatrixDataType& data) {
			int offset = matSizefrompos(*this, pos);
			*(m_ptrData.get() + offset) = data;
		}
		void CMatrix::setData(const std::vector<int>& pos, const MatrixDataType& data) {
			int offset = matSizefrompos(*this, pos);
			*(m_ptrData.get() + offset) = data;
		}
		MatrixDataType CMatrix::getData(const std::vector<int>& pos) const{
			int offset = matSizefrompos(*this, pos);
			return	*(m_ptrData.get() + offset);
		}

		MatrixDataType CMatrix::getData(const std::list<int>& pos) const{
			int offset = matSizefrompos(*this, pos);
			return	*(m_ptrData.get() + offset);
		}


		CMatrix CMatrix::linspace(const double& from, const double& to, const int& counts)
		{
			double step = (to - from) / (counts - 1.0);
			CMatrix tep_cache(list<int>{counts});//generate one dimension matrix
			for (int i = 0; i < counts - 1; ++i)
			{
				 tep_cache.m_ptrData[i] = from + i * step;
			}
			tep_cache.m_ptrData[counts - 1] = to;//fix the final one with to
			return tep_cache;
		}

		CMatrix::~CMatrix()
		{
			if (m_ptrData != nullptr)
			{
				m_ptrData = nullptr;
			}

		}

		bool matrixShapeEqual(const CMatrix& lhs, const CMatrix& rhs) {
			auto lhs_shape = lhs.shape();
			auto rhs_shape = lhs.shape();
			if (lhs_shape.size() != rhs_shape.size())return false;
			auto lhs_itr = lhs_shape.begin();
			auto rhs_itr = rhs_shape.begin();
			for (; lhs_itr != lhs_shape.end(); ++lhs_itr, ++rhs_itr) {
				if (*lhs_itr != *rhs_itr)return false;
			}
			return true;
		}






	}

}

