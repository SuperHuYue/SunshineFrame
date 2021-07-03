#pragma once
#include "CMatrix.hpp"
#include <cmath>
#include<iostream>
#include <functional>
#include <assert.h>
#include <initializer_list>
#include <iomanip>
namespace SunshineFrame {
	namespace Algebra {
		using namespace std;
		double const pi = 4 * std::atan(1.0);
		CMatrix::CMatrix(const std::initializer_list<int>& shape){
			std::list<int>listShape;
			for (auto i : shape) {
				listShape.push_back(i);
			}
			constructHelper(listShape);
		}
		CMatrix::CMatrix(const std::vector<int>& shape) {
			std::list<int>listShape;
			for (auto i : shape) {
				listShape.push_back(i);
			}
			constructHelper(listShape);
		}
		CMatrix::CMatrix(const std::list<int>& shape){
			std::list<int>listShape;
			for (auto i : shape) {
				listShape.push_back(i);
			}
			constructHelper(listShape);
		}
		void CMatrix::constructHelper(const std::list<int>& shape) {
			m_ptrData = nullptr;
			m_ndim = shape.size();
			m_nTotalSize = 1;
			bool bNegaApp = false;
			for (auto i = shape.begin(); i != shape.end(); ++i)
			{
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
			m_mapAxisCarryOver.clear();
			if (m_nTotalSize < 0)
			{
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
		bool CMatrix::broadcastRule(const std::list<int>& lhs, const std::list<int>& rhs,list<int>& out_shape)
		{
			out_shape.clear();
			auto it_lhs = lhs.rbegin();
			auto it_rhs = rhs.rbegin();
			do
			{
				if (it_lhs != lhs.rend() && it_rhs != rhs.rend())
				{
					if (*it_lhs > *it_rhs && 1 == *it_rhs) { out_shape.push_front(*it_lhs); it_lhs++; it_rhs++; continue; }
					if (*it_lhs < *it_rhs && 1 == *it_lhs) { out_shape.push_front(*it_rhs); it_lhs++; it_rhs++; continue; }
					if (*it_lhs == *it_rhs) { out_shape.push_front(*it_lhs); it_lhs++; it_rhs++; continue; };
					return false;
				}
				if (it_lhs == lhs.rend())
				{
					for (;it_rhs != rhs.rend(); it_rhs++)
					{
						out_shape.push_front(*it_rhs);
					}
					return true;
				}
				if (it_rhs == rhs.rend())
				{
					for (; it_lhs != lhs.rend(); it_lhs++)
					{
						out_shape.push_front(*it_lhs);
					}
					return true;
				}
				it_rhs++;
				it_lhs++;
			} while (true);
		}

		/*
		将in中的1 brocast到shape中的维度
		*/
		CMatrix CMatrix::genMatByBroadcastRule(const CMatrix& in, const std::list<int>& shape) {
			// CMatrix tmp_In = in;
			// std::list<int>inShape = tmp_In.m_listShape;
			// std::vector<int>vecFinalShape{shape.begin(), shape.end()};
			// if (inShape.size() != vecFinalShape.size()) {
			// 	int addDim = vecFinalShape.size() - inShape.size();
			// 	for (int i = 0; i < addDim; ++i) {
			// 		inShape.push_front(1);
			// 	}
			// }
			// tmp_In.reshape(inShape);
			// std::vector<int>vecInShape{ inShape.begin(), inShape.end() };
			// CMatrix outMat(shape);
			// auto inShapeItr = vecInShape.rbegin();
			// auto outShapeItr = shape.rbegin();
			// int stepSize = 1;
			// auto inMatDataPtr = tmp_In.getdataptr();
			// auto outMatDataPtr = outMat.getdataptr();
			/*
			此函数一次检测并broadcast一维的数据，如进行了broadcast则返回<true，新输出>没有则返回<false, 原输入>（代表与finshape参数一致）
			*/
			auto funcExpandForSingleAxis = [](const CMatrix& feedInMat, const std::vector<int>& finalShape)->std::pair<bool,SunshineFrame::Algebra::CMatrix>{
				auto feedInMatDataPtr = feedInMat.getdataptr();
				auto inShape = feedInMat.vecShape();
				auto outShape = inShape;
				if(inShape.size() != finalShape.size())throw std::runtime_error("FuncExpandForSingleAxis: check shape not match...");
				bool neHappend = false;
				int inShapeSize = inShape.size();

				int loopSize = 1;
				int times = 1;
				for(int i = inShapeSize - 1; i >=0; --i){
					if(!neHappend)loopSize*=inShape[i];
					if(finalShape[i] != inShape[i] &&
					   !neHappend){
						   if(inShape[i] == 1){
								neHappend = true;
								outShape[i] = finalShape[i];
								times = outShape[i];
						   }else{
							   throw std::runtime_error("funcExpandForsingleAxis err..., shape not equal and not one...");
						   }
					}
				}
				if(!neHappend)return std::make_pair(false, std::move(feedInMat));
				Algebra::CMatrix out(outShape);
				auto outMatDataPtr = out.getdataptr();
				int nCount = 0;
				for(int i = 0; i < feedInMat.m_nTotalSize; i+= loopSize){
					for(int sinLoop = 0; sinLoop <times; sinLoop++){
						memcpy(outMatDataPtr.get() + nCount, feedInMatDataPtr.get() + i, sizeof(Algebra::MatrixDataType) * loopSize);
						nCount += loopSize;
					/*	for(int innerIdx = i; innerIdx < i + loopSize; ++innerIdx){
							outMatDataPtr[nCount++] = feedInMatDataPtr[innerIdx];
						}*/
					}
				}
				return std::make_pair(true, std::move(out));
			};
			//保持shape纬度相同
			CMatrix tmp_In = in;
			std::list<int>inShape = tmp_In.m_listShape;
			std::vector<int>vecFinalShape{shape.begin(), shape.end()};
			if (inShape.size() != vecFinalShape.size()) {
				int addDim = vecFinalShape.size() - inShape.size();
				for (int i = 0; i < addDim; ++i) {
					inShape.push_front(1);
				}
			}
			tmp_In.reshape(inShape);
			//std::vector<int> in.vecShape();
			//std::vector<int>vecFinalShape{shape.begin(), shape.end()};
			int loopCount = 0;
			std::pair<bool,Algebra::CMatrix>out = funcExpandForSingleAxis(tmp_In,vecFinalShape);
			while(true){
				if(!out.first)break;
				out  = funcExpandForSingleAxis(out.second, vecFinalShape);
				if(loopCount++ >= vecFinalShape.size())throw std::runtime_error("Finite loop: something wrong in while funcexpandForSingleAxis...");
			}
			return out.second;

			//此方式速度奇慢
		/*	int offset = 0;
			for (int i = 0; i < outMat.m_nTotalSize; ++i) {
				auto vecPos = list2Vec(Algebra::CMatrix::matPosfromsize(outMat,i));
				for (auto j : notEqualVec) {
					vecPos[j] = 0;
				}
				offset = Algebra::CMatrix::matSizefrompos(tmp_In, vecPos);
				outMatDataPtr[i] = inMatDataPtr[offset];
			}
			return std::move(outMat);*/





			////检查size
			//int finalDim = shape.size();
			//int nowDim = in.m_listShape.size();
			//if (finalDim < nowDim)throw std::runtime_error("Err:genMatByBroadcastRule shut down, check broadcastRule, function have bugs... ");
			//auto expandFunc = [](const CMatrix& in, const std::list<int>& shape)->SunshineFrame::Algebra::CMatrix {
			//	//此函数为同维度下将shape为1的部分
			//	std::function<void(const CMatrix&, const std::list<int>&)>f;
			//	CMatrix finalMat;
			//	f = [&](const CMatrix& it, const std::list<int>& shape)->void{
			//		auto tmpShape = it.shape();
			//		if (tmpShape.size() != shape.size())throw std::runtime_error("Err:expandFunc shut down bugs in genMayByBroacastRule, please check...");
			//		auto itr_it = tmpShape.rbegin();
			//		int higestAxis = shape.size() - 1;
			//		int nowAxis = 0;
			//		CMatrix sinChangeOut;
			//		bool finish = true;
			//		for (auto itr_final = shape.rbegin(); itr_final != shape.rend(); ++itr_final,++itr_it, ++nowAxis) {
			//			if (*itr_it != *itr_final && *itr_it == 1) {
			//				auto needCopyObj = change_axis(it, nowAxis, higestAxis);
			//				auto tmp = needCopyObj.shape();
			//				if (tmp.front() != 1)throw std::runtime_error("Err:expandFunc shut down bugs in change_axis, please check...");
			//				tmp.pop_front();
			//				tmp.push_front(*itr_final);
			//				CMatrix target(tmp);
			//				for (int i = 0; i < *itr_final; ++i) {
			//					memcpy(target.getdataptr().get() + i * needCopyObj.gettotalsize(), needCopyObj.getdataptr().get(), needCopyObj.gettotalsize() * sizeof(MatrixDataType));
			//				}
			//				target = change_axis(target, higestAxis, nowAxis);
			//				sinChangeOut = target;
			//				finish = false;
			//				break;
			//			}
			//		}
			//		if (finish) {
			//			finalMat = it;
			//			return;
			//		}
			//		f(sinChangeOut, shape);
			//		return;
			//	};

			//	f(in, shape);
			//	return finalMat;
			//};
			//if (finalDim != nowDim) 
			//{
			//	//拷贝成维度相等内容
			//	int needCopySize = in.gettotalsize();
			//	int needCopyTimes = 1;
			//	std::list<int>tmp_shape = in.m_listShape;
			//	int count = 1;
			//	for (auto itr = shape.rbegin(); itr != shape.rend(); ++itr,++count) {
			//		if (count <= nowDim)continue;
			//		tmp_shape.push_front(*itr);
			//		needCopyTimes *= *itr;
			//	}
			//	CMatrix tmp(tmp_shape);
			//	for (int i = 0; i < needCopyTimes; ++i) {
			//		memcpy(tmp.getdataptr().get() + (size_t)i * needCopySize, in.getdataptr().get(), needCopySize * sizeof(MatrixDataType));
			//	}
			//	return expandFunc(tmp, shape);
			//}
			//else 
			//{
			//	return expandFunc(in, shape);

			//}
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
			bool ok = CMatrix::broadcastRule((*this).m_listShape, rhs.m_listShape, out_shape);
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
			bool ok = CMatrix::broadcastRule((*this).m_listShape, rhs.m_listShape, out_shape);
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
			bool ok = CMatrix::broadcastRule((*this).m_listShape, rhs.m_listShape, out_shape);
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
		*因此，对于高于二维的维度每一个都必须相同(或者没或者1)，遍历出来的二维矩阵需要满足向量乘法法则
		*/
		CMatrix CMatrix::matmul(const CMatrix& lhs, const CMatrix& rhs)
		{
			std::vector<int>lhs_shape{ lhs.m_listShape.begin(), lhs.m_listShape.end() };
			std::vector<int>rhs_shape{ rhs.m_listShape.begin(), rhs.m_listShape.end() };
			int lhsShapeSize = lhs_shape.size();
			int rhsShapeSize = rhs_shape.size();
			if (lhsShapeSize == 1 || rhsShapeSize == 1)throw std::runtime_error("Err: matmul err reason dim can't be one...");
			if (lhs_shape[lhsShapeSize - 1] != rhs_shape[rhsShapeSize - 2]) throw std::runtime_error("Err: matmul lhs ,rhs shape can't do matmul..");
			//高于2的维度使用broadcast规则
			int shapeMax = lhsShapeSize > rhsShapeSize ? lhsShapeSize : rhsShapeSize;
			auto lhsOut = lhs;
			auto rhsOut = rhs;
			if (shapeMax > 2) {
				std::list<int>tmpLhsShapeBC;
				std::list<int>tmpRhsShapeBC;
				for (int i = 0; i < lhsShapeSize - 2; ++i) {
					tmpLhsShapeBC.push_back(lhs_shape[i]);
				}
				for (int i = 0; i < rhsShapeSize - 2; ++i) {
					tmpRhsShapeBC.push_back(rhs_shape[i]);
				}
				std::list<int>finalShape;
				if (!broadcastRule(tmpLhsShapeBC, tmpRhsShapeBC, finalShape)) {
					throw std::runtime_error("Err: matmul can't do matmul in bc...");
				}
				std::list<int>lhsFinalShape = finalShape;
				std::list<int>rhsFinalShape = finalShape;
				lhsFinalShape.push_back(lhs_shape[lhsShapeSize - 2]);
				lhsFinalShape.push_back(lhs_shape[lhsShapeSize - 1]);
				rhsFinalShape.push_back(rhs_shape[rhsShapeSize - 2]);
				rhsFinalShape.push_back(rhs_shape[rhsShapeSize - 1]);

			//	finalShape.push_back(lhs_shape[lhsShapeSize - 2]);
			//	finalShape.push_back(lhs_shape[rhsShapeSize - 1]);
				lhsOut = genMatByBroadcastRule(lhs, lhsFinalShape);
				rhsOut =genMatByBroadcastRule(rhs, rhsFinalShape);
			}
			////////////////////////////////////////////////////////////////////////////////
			//高维支持
			auto rhsVecShape =  rhsOut.vecShape();
			auto lhsVecShape = lhsOut.vecShape();
			int rhsVecRowIdx = rhsVecShape.size() - 2;
			int rhsVecColIdx = rhsVecShape.size() - 1;
			int lhsVecRowIdx = lhsVecShape.size() - 2;
			int lhsvecColIdx = lhsVecShape.size() - 1;

			int times = 1;
			int lhsFloorSize = lhsVecShape[lhsvecColIdx] * lhsVecShape[lhsVecRowIdx];
			int lhsFloorOffset = 0;
			int lhsRowSizeOffset = 0;
			int rhsFloorSize = rhsVecShape[rhsVecColIdx] * rhsVecShape[rhsVecRowIdx];
			int rhsRowSizeOffset = 0;
			int rhsFloorOffset = 0;
			std::list<int>outShape;
			for (int i = 0; i < lhsVecShape.size() - 2; ++i) {
				times *= lhsVecShape[i];
				outShape.push_back(lhsVecShape[i]);
			}
			int outFloorSize = lhsVecShape[lhsVecRowIdx] * rhsVecShape[rhsVecColIdx];
			outShape.push_back(lhsVecShape[lhsVecRowIdx]);
			outShape.push_back(rhsVecShape[rhsVecColIdx]);
			auto lhsDataPtr = lhsOut.getdataptr();
			auto rhsDataPtr = rhsOut.getdataptr();
			CMatrix outMat(outShape);
			auto outMatDataPtr = outMat.getdataptr();
			int count = 0;
			int data = 0;
			for (int i = 0; i < times; ++i) {
				for (int or = 0; or < lhsVecShape[lhsVecRowIdx]; ++or ) {
					for (int oc = 0; oc < rhsVecShape[rhsVecColIdx]; ++oc) {
						data = 0;
						for (int innerLoop = 0; innerLoop < lhsVecShape[lhsvecColIdx]; ++innerLoop) {
							//此处可以加速
							int left_data_idx = i * lhsFloorSize + or *lhsVecShape[lhsvecColIdx] + innerLoop;
							int right_data_idx = i * rhsFloorSize + oc + innerLoop * rhsVecShape[rhsVecColIdx];
							data += lhsDataPtr[left_data_idx] * rhsDataPtr[right_data_idx];
						}
						outMatDataPtr[count++] = data;
					}
				}
			}
			return std::move(outMat);
			


			////////////////////////////////////////////////////////////////////////////////
			//高维支持
			//int tep_count = 1;
			//int times = 1;//开辟内存倍数
			//int single_mul_size_out = 1;//单个矩阵乘法输出需要偏移的大小
			//int single_mul_size_rhs = 1;//单个矩阵乘法rhsOut需要偏移的大小
			//int single_mul_size_lhs = 1;//单个矩阵乘法lhsOut需要偏移的大小
			//list<int> out_shape;//eg:[6;7;4] * [6;4;5]====>[6;7;5]
			//auto it_rhs = rhsOut.m_listShape.rbegin();
			//int two_dim_row = 1;
			//int two_dim_col = 1;
			//for (auto it_lhs = lhsOut.m_listShape.rbegin(); it_lhs != lhsOut.m_listShape.rend(); ++it_lhs,++it_rhs,++tep_count)
			//{
			//	if (tep_count == 1) { 
			//		out_shape.push_front(*it_rhs);
			//		two_dim_col = *it_rhs;
			//		single_mul_size_out *= *it_rhs;
			//		single_mul_size_lhs *= *it_lhs;
			//		single_mul_size_rhs *= *it_rhs;
			//	}
			//	else if (tep_count == 2) {
			//		out_shape.push_front(*it_lhs);
			//		two_dim_row = *it_lhs;
			//		single_mul_size_out *= *it_lhs;
			//		single_mul_size_lhs *= *it_lhs;
			//		single_mul_size_rhs *= *it_rhs;
			//	}
			//	else {
			//		out_shape.push_front(*it_rhs);
			//	}
			//}
			//times = lhsOut.m_nTotalSize / single_mul_size_lhs;

			//CMatrix outMatrix(out_shape);
			//LONGLONG outtotalsize = outMatrix.gettotalsize();
			//LONGLONG lhstotalsize = lhsOut.gettotalsize();
			//LONGLONG rhstotalsize = rhsOut.gettotalsize();
			//auto ptr_out = outMatrix.getdataptr();
			//auto ptr_lhs = lhsOut.getdataptr();
			//auto ptr_rhs = rhsOut.getdataptr();

			//int feed_col = lhsOut.m_listShape.back();		
			//for (int i = 0; i < times; i++)
			//{
			//	for (int index_row = 0; index_row < two_dim_row; ++index_row)
			//	{
			//		for (int index_col = 0; index_col < two_dim_col; index_col++)
			//		{
			//			//进行计算
			//			MatrixDataType tep_total = 0;
			//			assert(i * (size_t)single_mul_size_out + (size_t)index_row * outMatrix.m_listShape.back() + index_col < outtotalsize);//out越界控制
			//			for (int j = 0; j < lhsOut.m_listShape.back(); ++j)
			//			{
			//				
			//				assert(i * (size_t)single_mul_size_lhs + (size_t)index_row * lhsOut.m_listShape.back() + j < lhstotalsize);//lhsOut越界控制
			//				assert(i * (size_t)single_mul_size_rhs + (size_t)j * rhsOut.m_listShape.back() + index_col < rhstotalsize);//rhsOut越界控制
			//				Algebra::MatrixDataType a = ptr_lhs[i * single_mul_size_lhs + (size_t)index_row * lhsOut.m_listShape.back() + j];
			//				Algebra::MatrixDataType b = ptr_rhs[i * single_mul_size_rhs + (size_t)j * rhsOut.m_listShape.back() + index_col];
			//				tep_total +=
			//					ptr_lhs[i * single_mul_size_lhs + (size_t)index_row * lhsOut.m_listShape.back() + j] *
			//					ptr_rhs[i * single_mul_size_rhs + (size_t)j * rhsOut.m_listShape.back() + index_col];
			//			}
			//			ptr_out[i * single_mul_size_out + (size_t)index_row * outMatrix.m_listShape.back() + index_col] = tep_total;
			//		}

			//	}
			//}
			//return outMatrix;
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
			if (from == to) { return enter; };
			int maxAxis = from > to ? from : to;
			if (maxAxis > enter.m_ndim)throw std::runtime_error("change_Axis: axis bigger than enter dim..");
			auto vecEnterShape = enter.vecShape();
			int tmpCache = vecEnterShape[from];
			vecEnterShape[from] = vecEnterShape[to];
			vecEnterShape[to] = tmpCache;
			Algebra::CMatrix outMat(vecEnterShape);
			auto outMatDataPtr = outMat.getdataptr();
			auto enterMatDataPtr = enter.getdataptr();
			for (int i = 0; i < enter.m_nTotalSize; ++i) {
				auto pos = list2Vec(matPosfromsize(enter, i));
				tmpCache = pos[from];
				pos[from] = pos[to];
				pos[to] = tmpCache;
				auto offset = matSizefrompos(outMat, pos);
				outMatDataPtr[offset] = enterMatDataPtr[i];
			}
			return std::move(outMat);
			////加速
			//if (from == to) {
			//	return enter;
			//}
			//int max_axis_num = enter.m_ndim - 1;
			//assert(max_axis_num >= from && max_axis_num >= to && from >= 0 && to >= 0);
			//list<int> enter_cache = enter.m_listShape;
			//{
			//	int tep_count = 0;
			//	int a_from = 0;
			//	int b_to = 0;
			//	auto a_idx = enter_cache.rbegin();
			//	auto b_idx = enter_cache.rbegin();
			//	for (auto it = enter_cache.rbegin(); it != enter_cache.rend(); ++it, ++tep_count)
			//	{
			//		if (tep_count == from) {
			//			a_from = *it;
			//			a_idx = it;
			//		}
			//		if (tep_count == to) {
			//			b_to = *it;
			//			b_idx = it;
			//		}
			//	}
			//	*a_idx = b_to;
			//	*b_idx = a_from;
			//}
			//CMatrix output(enter_cache);

			//list<int> pos;
			//for (int i = 0; i < enter.m_nTotalSize; ++i)
			//{
			//	list<int>tep_from = matPosfromsize(enter, i);
			//	MatrixDataType ori_data = enter.m_ptrData[i];
			//	int axis_count = 0;
			//	int from_axis_data = 0;
			//	int to_axis_data = 0;
			//	auto it_from_idx = tep_from.rbegin();
			//	auto it_to_idx = tep_from.rbegin();
			//	for (auto it = tep_from.rbegin(); it != tep_from.rend(); ++it, ++axis_count)
			//	{
			//		if (axis_count == from) {
			//			from_axis_data = *it;
			//			it_from_idx = it;
			//		}
			//		if (axis_count == to) {
			//			to_axis_data = *it;
			//			it_to_idx = it;
			//		}
			//	}
			//	*it_from_idx = to_axis_data;
			//	*it_to_idx = from_axis_data;		
			//	MatrixDataType* tar_pos_data = output.getPosDataPtr(tep_from);
			//	*tar_pos_data = ori_data;
			//}
			////output.reshape(enter_cache);
			//return output;
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
			return std::move(pos);
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

