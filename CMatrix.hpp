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
		* ��matrix�ڴ�ṹ ex��5��3��4��---5�3�� 4��
		*��axis��:ex����5��3��4�� axis = 0��ӦΪ4��axis = 1��ӦΪ3��axis = 2��ӦΪ5(��numpy�궨�෴)
		*�ڴ�ռ�洢����:
		    ��������������[[1,2,3],
						   [4,5,6]]   ��Ϊһ��2*3�ľ����������ǵ��ڴ�ֲ���Ϊ1,2,3,4,5,6;��ά�������

		*ע�⣺1.�������������ж�����MatrixDataType	
		*     2.���Դ���һ���յ�shape�������
		*     3.Ĭ������Ϊ������CMatrix a(list<int>{5})---Ϊһ�У����������Ԫ�� 
		*/
		using MatrixDataType = long double;
		class CMatrix
		{
		public:
			/*
			*���ܣ���ʼ������ֵΪ0
			*shape:������״��Ϣ��ex:list�д����3��2��ôaxis =0 ������Ԫ�أ�axis =1 ������Ԫ�ء����� ��numpyһ��
			*dtype�����ݴ洢�ĸ�ʽ
			*Ϊ�˼��٣�����֤��������һ���������ڴ�֮��
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
			void matrixFeed(std::list<MatrixDataType>data);//�����ڴ�ṹ�������и�ֵ

			bool reshape(std::list<int>shape);
			inline std::list<int> shape() const{ return m_listShape; };
			inline std::map<int, int> getAxisCarryOver() const{ return m_mapAxisCarryOver; };
			void print()const;//չʾ�����ڲ�����
			void setData(const std::vector<int>& pos, const MatrixDataType& data);
			void setData(const std::list<int>& pos, const MatrixDataType& data);
			MatrixDataType getData(const std::list<int>& pos)const;
			MatrixDataType getData(const std::vector<int>& pos)const;

		public:
			inline void CalAxisCarry();//m_mapAxisCaryyOver�������
			inline std::shared_ptr<MatrixDataType[]> getdataptr()const { return m_ptrData; };
			inline const int& gettotalsize()const { return m_nTotalSize; };
			MatrixDataType* getPosDataPtr(std::list<int>pos);
			MatrixDataType getPosData(std::list<int> pos);
			void zeros();
			void ones();
			void fixNumbers(const MatrixDataType& num);

			/*
			*box_muller������̫�ֲ�����
			*mu:��ֵ
			*sigma������
			*seed:���������
			*/
			void random_normalize(const double& mu, const double& sigma, int seed = -1);

			CMatrix T()const;//ת��
		public://��̬����
			static CMatrix matmul(const CMatrix& lhs, const CMatrix& rhs);
			static CMatrix linspace(const double& from, const double& to, const int& counts);
			static CMatrix zeros(std::list<int> shape);
			static CMatrix ones(std::list<int> shape);
			/*�㲥���� 1. ���������ά�ȴ�С�Ӻ���ǰ�ȶԾ�һ��
			             2. �����������һЩά�ȴ�С�����ʱ����һ������ĸò����ά�ȴ�СΪ1
			broadcast rule:�ж��㲥�����Ƿ�����
			*bool:true���Ϲ㲥����false �����Ϲ㲥����
			*lhs:��߲�������
			*rhs���ұ߲�������
			*out_size���ڷ���ֵΪtrue�������Ӧ������ĳߴ�
			*/
			static bool broadcastRule(const CMatrix& lhs, const CMatrix& rhs, std::list<int>& out_shape);

			/*
			����broadcastRule���ɵĳߴ�����루in���ľ������broadcast
			return value: broadcast֮������shape�ߴ�ľ���
			*/
			static CMatrix genMatByBroadcastRule(const CMatrix& in, const std::list<int>& shape);

			//�����������ԭ������from��ŵ�����ת����to����У���reshape�����ı��ǩ�ǲ�ͬ��
			//ex:����ԭ�ȳߴ�Ϊ[5,3,2] ����axisFrom=1,axisTo=0��ߴ��Ϊ[5,2,3],��reshape��ͬ�ĵط�����ԭ��[2,1,3]�����ݱ任���ڴ�λ�ã����ڵ�ͬ��[2,3,1]�е����� 
			static CMatrix change_axis(const CMatrix& enter, const int& axisFrom, const int& axisTo);
			//�����ڴ��еĸ����ж���Ӧ�����е�index
			//size:���ٿռ�ĵڼ���Ԫ��
			static std::list<int> matPosfromsize(const CMatrix& enter, const int& size);
			//����pos ָʾ��λ�÷����ڴ�����׵�ַ��ƫ����: eg: pos=(1,2)  ��3*3 �ľ����н��᷵��5
			static int matSizefrompos(const CMatrix& enter, const std::list<int>& pos);
			//����pos ָʾ��λ�÷����ڴ�����׵�ַ��ƫ����: eg: pos=(1,2)  ��3*3 �ľ����н��᷵��5
			static int matSizefrompos(const CMatrix& enter, const std::vector<int>& pos);
			/* ȡ��ֵ����
			* ��axis��Ӧ�����ȡ����ȡ��ֵ�Ĳ���
			* keepdim:�������������enterͬγ��,��������1������̮��ex: (5,1,3) ==> (5,3)
			*/
			static CMatrix mean(const CMatrix& enter, const int& axis, const bool& keepdim = true);

		private:
			void easy_changeshape(std::list<int> shape);//change shape and the m_ndim 
			bool cekIdxOk(std::list<int>pos);//check index reasonable
			std::list<int> m_listShape;//����ĳߴ�,��ͨ��reshape����
			//���λmap����һ������������ţ��ڶ���������������������ٸ�������byte��ӦΪ�������ﶼ��MatrixDataType�����ܵ������ά���е���һ��Ԫ�أ����shape����-1��ô�˲�������ʹ��ͬʱmap.size()����Ϊ0
			std::map<int, int>m_mapAxisCarryOver;
		//	double* m_ptrData;//����ָ�룬����ͨ��getdataptrֱ�ӽ��в�������Ҫ����
			std::shared_ptr<MatrixDataType[]> m_ptrData;
			
			int m_nTotalSize;//�ܹ���ռ�ݵĿռ�
			int m_ndim;//����ά��
		};
		//broad cast rule one:
		CMatrix operator*(const MatrixDataType& lhs, const CMatrix& rhs);
		CMatrix operator*(const int& lhs, const CMatrix& rhs);
		CMatrix operator+(const MatrixDataType& lhs, const CMatrix& rhs);
		//
		bool matrixShapeEqual(const CMatrix& lhs, const CMatrix& rhs);
	}
}
