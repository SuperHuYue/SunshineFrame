#pragma once
#include "baselayer.hpp"
namespace SunshineFrame {
	namespace Layer {
		class FullyConnectLayer : public SunshineBaseLayer
		{
			const char SPLITCHAR = ',';
		public:
			FullyConnectLayer() = delete;
			/*
				//全连接层
				//shape节点的个数 + weight的个数 +  eg:3*2 代表 y= w2 * x2 + w1 * x1 + b0 (一共有2个不同w), b0为bias,3代表有两个这样的元node
				*/
			FullyConnectLayer(const std::list<int> shape, const double &learnRate = 0.005, const std::string user_alias = "undefined", bool activate = true, bool freezen = false)
				: SunshineBaseLayer(LayerType::FullyConnect, learnRate, user_alias, activate, freezen)
			{
				m_weightMat = Algebra::CMatrix(shape);
				m_biasMat = Algebra::CMatrix::ones({shape.front(), 1});
				m_weightMat.random_normalize(0, 1);
				m_biasMat.random_normalize(0, 1);
			};
			~FullyConnectLayer(){};
			void weightMatFeed( std::list<Algebra::MatrixDataType> data)
			{
				m_weightMat.matrixFeed(data);
			}
			void biasMatFeed( std::list<Algebra::MatrixDataType> data)
			{
				m_biasMat.matrixFeed(data);
			}

		private:
			void loadPara(std::string& para) override {
				SunshineBaseLayer::loadPara(para);
				for (int i = 0; i < m_weightMat.gettotalsize(); ++i) {
					std::string&& data = findValue(para, SPLITCHAR);
					Algebra::MatrixDataType a = std::strtod(data.c_str(), nullptr);
					*(m_weightMat.getdataptr().get() + i) = std::strtod(data.c_str(),nullptr);
				}
				for (int i = 0; i < m_biasMat.gettotalsize(); ++i) {
					std::string&& data = findValue(para, SPLITCHAR);
					Algebra::MatrixDataType a = std::strtod(data.c_str(), nullptr);
					*(m_biasMat.getdataptr().get() + i) = std::strtod(data.c_str(),nullptr);
				}
			};
			void savePara(std::ostream& ofs)override {
				SunshineBaseLayer::savePara(ofs);
				ofs << std::setprecision(std::numeric_limits<Algebra::MatrixDataType>::max_digits10);
				for (int i = 0; i < m_weightMat.gettotalsize(); ++i) {
					Algebra::MatrixDataType a = *(m_weightMat.getdataptr().get() + i);
					ofs << *(m_weightMat.getdataptr().get() + i) << ",";
				}
				for (int i = 0; i < m_biasMat.gettotalsize(); ++i) {
					Algebra::MatrixDataType a = *(m_biasMat.getdataptr().get() + i);
					ofs << *(m_biasMat.getdataptr().get() + i) << ",";
				}
				ofs << std::setprecision(6);
			};
			void calBackUpdateMat(const Algebra::CMatrix &loss) override
			{
				m_backUpdateBiasMat = Algebra::CMatrix::ones({m_backUpdateMat.shape().back(), 1});
				m_backUpdateMat = Algebra::CMatrix::matmul(loss, m_backUpdateMat.T()) * m_layerLearningRate;
				m_back2frontMat = Algebra::CMatrix::matmul(m_weightMat.T(), loss);
				m_backUpdateBiasMat = Algebra::CMatrix::matmul(loss, m_backUpdateBiasMat) * m_layerLearningRate;
				return;
			};
			void updateBackMat(const Algebra::CMatrix &loss) override
			{
				SunshineBaseLayer::updateBackMat(loss);
				m_weightMat = m_weightMat - m_backUpdateMat;
				m_biasMat = m_biasMat - m_backUpdateBiasMat;
				return;
			};
			void showWeight() override{
				SunshineBaseLayer::showWeight();
				std::cout << "------------------weight---------------\n";
				m_weightMat.print();
				std::cout << "------------------bias---------------\n";
				m_biasMat.print();
				std::cout << "------------------updateWeightMat---------------\n";
				m_backUpdateMat.print();
				std::cout << "------------------updateBiasMat---------------\n";
				m_backUpdateBiasMat.print();

			}

			void forwardMove(const Algebra::CMatrix &feedIn) override
			{
				SunshineBaseLayer::forwardMove(feedIn);
				m_front2backMat = Algebra::CMatrix::matmul(m_weightMat, feedIn) + m_biasMat;
				m_backUpdateMat = feedIn;
				return;
			};
			void forwardMove() override
			{
				assert(m_frontConnectLayer.size() == 1);
				SunshineBaseLayer::forwardMove();
				auto layer = m_frontConnectLayer.front();
				m_front2backMat = Algebra::CMatrix::matmul(m_weightMat, layer->getFront2BackMat()) + m_biasMat;
				m_backUpdateMat = layer->getFront2BackMat();
				return;
			}
			void backwardMove() override
			{
				//assert(m_BackConnectLayer.size() == 1);
				SunshineBaseLayer::backwardMove();
				for (auto layerItr = m_BackConnectLayer.begin(); layerItr != m_BackConnectLayer.end(); ++layerItr) {
					//auto loss = layer->getBack2FrontMat();
					auto loss = (*layerItr)->getBack2FrontMat();
					calBackUpdateMat(loss);
					updateBackMat(loss);
				}
				return;
			}

		private:
			/* 参数矩阵,单个节点系数使用列方式进列排列,多节点行堆叠
				ex:
					A= [[1,2,3]
						[4,5,6]]
					代表为: y0 = 1 * x0 + 2 * x1 + 3*x2;
							y1 = 4 * x0 + 5 * x1 + 6*x2;
					此式子中一共有两个node，输入数据为三维数据
				*/
			Algebra::CMatrix m_weightMat;
			Algebra::CMatrix m_biasMat;	      //偏置矩阵代表b,使用m_weightMat的数据加上b可获得最终输出
			Algebra::CMatrix m_backUpdateMat;     //后向传播更新参数矩阵, 用于更新m_weightmat参数矩阵
			Algebra::CMatrix m_backUpdateBiasMat; //后向传播更新参数矩阵, 用于更新m_biasMat参数矩阵,在fullyconnect中为定值（全1矩阵）
		};

	}

}