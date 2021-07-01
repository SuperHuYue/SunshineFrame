#pragma once
#include <memory>
#include<string>
#include<assert.h>
#include <vector>
#include <list>
#include <map>
namespace SunshineFrame {
	namespace Algebra{
		/*
		* 本matrix内存结构 ex（5，3，4）---5深，3行 4列
		*，axis号:ex：（5，3，4） axis = 0对应为4，axis = 1对应为3，axis = 2对应为5(与numpy标定相反)
		*内存空间存储布局:
		    假设有如下数据[[1,2,3],
						   [4,5,6]]   其为一个2*3的矩阵，其在我们的内存分布中为1,2,3,4,5,6;高维度亦如此

		*注意：1.操作函数必须判定类型MatrixDataType	
		*     2.可以传入一个空的shape代表标量
		*     3.默认输入为行向量CMatrix a(list<int>{5})---为一行，其中有五个元素 
		*/
		using MatrixDataType = long double;
		class CMatrix
		{
		public:
			/*
			*功能：初始化并赋值为0
			*shape:矩阵形状信息，ex:list中存的是3，2那么axis =0 有两个元素，axis =1 有三个元素。。。 与numpy一样
			*dtype：数据存储的格式
			*为了加速，将举证放置在了一串连续的内存之上
			*
			*/
			CMatrix() :CMatrix({ 1,1 }) {};
			CMatrix(const std::list<int>& shape);
			~CMatrix();
			CMatrix(const CMatrix& cpy);
			CMatrix& operator=(const CMatrix& rhs);
			CMatrix operator*(const CMatrix& rhs)const;
			CMatrix operator+(const CMatrix& rhs);
			CMatrix operator-(const CMatrix& rhs);
			CMatrix& operator+=(const CMatrix& rhs);
			CMatrix& operator-=(const CMatrix& rhs);
			CMatrix& operator*=(const CMatrix& rhs);
			bool operator==(const CMatrix& rhs)const;
			CMatrix operator+(const MatrixDataType& rhs);
			CMatrix operator+(const int& rhs);
			CMatrix operator*(const MatrixDataType& rhs);
			CMatrix operator*(const int& rhs);

			CMatrix& operator=(CMatrix &&rhs)noexcept;
			CMatrix(CMatrix &&cpy)noexcept;
			void matrixFeed(std::list<MatrixDataType>data);//按照内存结构进行排列赋值

			bool reshape(std::list<int>shape);
			inline std::list<int> shape() const{ return m_listShape; };
			inline std::map<int, int> getAxisCarryOver() const{ return m_mapAxisCarryOver; };
			void print()const;//展示自身内部数据
			void setData(const std::vector<int>& pos, const MatrixDataType& data);
			void setData(const std::list<int>& pos, const MatrixDataType& data);
			MatrixDataType getData(const std::list<int>& pos)const;
			MatrixDataType getData(const std::vector<int>& pos)const;

		public:
			inline void CalAxisCarry();//m_mapAxisCaryyOver计算参数
			inline std::shared_ptr<MatrixDataType[]> getdataptr()const { return m_ptrData; };
			inline const int& gettotalsize()const { return m_nTotalSize; };
			MatrixDataType* getPosDataPtr(std::list<int>pos);
			MatrixDataType getPosData(std::list<int> pos);
			void zeros();
			void ones();
			void fixNumbers(const MatrixDataType& num);

			/*
			*box_muller生成正太分布序列
			*mu:均值
			*sigma：方差
			*seed:随机数种子
			*/
			void random_normalize(const double& mu, const double& sigma, int seed = -1);

			CMatrix T()const;//转置
		public://静态函数
			static CMatrix matmul(const CMatrix& lhs, const CMatrix& rhs);
			static CMatrix linspace(const double& from, const double& to, const int& counts);
			static CMatrix zeros(std::list<int> shape);
			static CMatrix ones(std::list<int> shape);
			/*广播规则： 1. 两个数组各维度大小从后往前比对均一致
			             2. 两个数组存在一些维度大小不相等时，有一个数组的该不相等维度大小为1
			broadcast rule:判定广播规则是否满足
			*bool:true符合广播规则。false 不符合广播规则
			*lhs:左边操作矩阵
			*rhs：右边操作矩阵
			*out_size：在返回值为true的情况下应该输出的尺寸
			*/
			static bool broadcastRule(const CMatrix& lhs, const CMatrix& rhs, std::list<int>& out_shape);

			/*
			依据broadcastRule生成的尺寸对输入（in）的矩阵进行broadcast
			return value: broadcast之后满足shape尺寸的矩阵
			*/
			static CMatrix genMatByBroadcastRule(const CMatrix& in, const std::list<int>& shape);

			//变轴操作：将原先属于from轴号的内容转换到to轴号中，和reshape单纯的变标签是不同的
			//ex:假设原先尺寸为[5,3,2] 进行axisFrom=1,axisTo=0则尺寸变为[5,2,3],与reshape不同的地方在于原先[2,1,3]的数据变换了内存位置，现在等同于[2,3,1]中的数据 
			static CMatrix change_axis(const CMatrix& enter, const int& axisFrom, const int& axisTo);
			//根据内存中的个数判定对应矩阵中的index
			//size:开辟空间的第几个元素
			static std::list<int> matPosfromsize(const CMatrix& enter, const int& size);
			//根据pos 指示的位置返回内存距离首地址的偏移量: eg: pos=(1,2)  在3*3 的矩阵中将会返回5
			static int matSizefrompos(const CMatrix& enter, const std::list<int>& pos);
			//根据pos 指示的位置返回内存距离首地址的偏移量: eg: pos=(1,2)  在3*3 的矩阵中将会返回5
			static int matSizefrompos(const CMatrix& enter, const std::vector<int>& pos);
			/* 取均值操作
			* 对axis对应的轴号取进行取均值的操作
			* keepdim:输出保持与输入enter同纬度,否则遇到1则会进行坍缩ex: (5,1,3) ==> (5,3)
			*/
			static CMatrix mean(const CMatrix& enter, const int& axis, const bool& keepdim = true);

		private:
			void easy_changeshape(std::list<int> shape);//change shape and the m_ndim 
			bool cekIdxOk(std::list<int>pos);//check index reasonable
			std::list<int> m_listShape;//矩阵的尺寸,需通过reshape更改
			//轴进位map，第一个参数代表轴号，第二个参数代表必须跳过多少个数（非byte，应为我们这里都是MatrixDataType）才能到达该轴维度中的下一个元素，如果shape存在-1那么此参数不能使用同时map.size()将会为0
			std::map<int, int>m_mapAxisCarryOver;
		//	double* m_ptrData;//数据指针，可以通过getdataptr直接进行操作，需要谨慎
			std::shared_ptr<MatrixDataType[]> m_ptrData;
			
			int m_nTotalSize;//总共所占据的空间
			int m_ndim;//数据维数
		};
		//broad cast rule one:
		CMatrix operator*(const MatrixDataType& lhs, const CMatrix& rhs);
		CMatrix operator*(const int& lhs, const CMatrix& rhs);
		CMatrix operator+(const MatrixDataType& lhs, const CMatrix& rhs);
		//
		bool matrixShapeEqual(const CMatrix& lhs, const CMatrix& rhs);
	}
}
