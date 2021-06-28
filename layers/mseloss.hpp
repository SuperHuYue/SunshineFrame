#pragma once
#include "baselayer.hpp"

namespace SunshineFrame {
	namespace Layer {
		class MseLossLayer : public SunshineBaseLayer
		{
		public:
			/*
				Mse:mean-square error
				*/
			MseLossLayer(bool activate = true, bool freezen = false, std::string userAlias = "undefined")
				: SunshineBaseLayer(LayerType::LossLayer, 0.01, userAlias, activate, freezen)
			{
				m_lossLayerType = LossLayerType::MSE;
			};
			~MseLossLayer(){};

		private:
			/*
				函数会主动像前向的节点的front2back索要数据 
				feedIn:标记好了的真实标签
				*/
			void forwardMove(const Algebra::CMatrix &feedIn) override
			{
				SunshineBaseLayer::forwardMove(feedIn);
				std::string errHeader = "MesLossLayer: " + lossID2String() + " ";
				if (m_frontConnectLayer.size() > 2)
					throw std::runtime_error(errHeader + "  reason: frontConnectsize bigger than tow..");
				auto predictOut = m_frontConnectLayer.front()->m_front2backMat;
				bool shapeEqual = Algebra::matrixShapeEqual(predictOut, feedIn);
				if (!shapeEqual)
					throw std::runtime_error(errHeader + " reason: shapeNotEuqal...");
				Algebra::CMatrix out = (predictOut - feedIn);
				m_back2frontMat = out * 2;
				out = out * out;
				m_front2backMat = Algebra::CMatrix::mean(out, 1);
				m_front2backMat = Algebra::CMatrix::mean(m_front2backMat, 0); //最后的损失函数
				//std::cout << "now loss is : \n";
				m_front2backMat.print();
				return;
			};
			void backwardMove() override
			{
				SunshineBaseLayer::backwardMove();
				m_back2frontMat = m_back2frontMat; // *m_front2backMat;
				return;
			}
		};

	}
}
