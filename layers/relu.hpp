#pragma once
#include "baselayer.hpp"
namespace SunshineFrame 
{
	namespace Layer 
	{
		class ReluLayer : public SunshineBaseLayer
		{
		public:
			/*
				Relu
				*/
			ReluLayer(bool activate = true, bool freezen = false, std::string userName = "undefined")
				: SunshineBaseLayer(LayerType::Relu, 0.01,userName, activate, freezen)
			{
				m_lossLayerType = LossLayerType::Undefined;
			};
			~ReluLayer(){};

		private:
			/*
				函数会主动像前向的节点的front2back索要数据
				*/
			void forwardMove() override
			{
				SunshineBaseLayer::forwardMove();
				m_front2backMat = m_frontConnectLayer.front()->m_front2backMat;
				m_back2frontMat = m_front2backMat;
				auto ptr = m_front2backMat.getdataptr();
				auto backPtr = m_back2frontMat.getdataptr();
				for (int i = 0; i < m_front2backMat.gettotalsize(); ++i)
				{
					*(ptr.get() + i) = *(ptr.get() + i) > 0 ? *(ptr.get() + i) : 0;
					*(backPtr.get() + i) = *(ptr.get() + i) > 0 ? 1 : 0;
				}
				//m_back2frontMat = Algebra::CMatrix::mean(m_front2backMat, 0);
				//for (int i = 0; i < m_back2frontMat.gettotalsize(); ++i) {
				//	*(m_back2frontMat.getdataptr().get() + i) = *(m_back2frontMat.getdataptr().get() + i) > 0 ? 1 : 0;
				//}
				return;
			};
			void backwardMove() override
			{
				SunshineBaseLayer::backwardMove();
				m_back2frontMat *= m_BackConnectLayer.front()->m_back2frontMat;
				return;
			}
		};

	}

}