#pragma once
#include "baselayer.hpp"
/*
�����Ŀǰ��δ��ɣ��Ϲ��ӹ���
*/
namespace SunshineFrame {
	namespace Layer {
		class ConvLayer2D: public SunshineBaseLayer
		{
			const char SPLITCHAR = ',';
		public:
			ConvLayer2D() = delete;
			/*
				�����:
				filterSize:�˵ĸ���
				shape:��У��� ����CMatrixһ�£�
				padding: no--- nopadding, same---��δʵ��,
				step:���ƶ����������ƶ�����
			*/
			ConvLayer2D(const int filterSize,const std::list<int> shape, const std::string padding = "noPadding", const std::pair<int, int> step = {1,1}, const double& learnRate = 0.005, const std::string user_alias = "undefined", bool activate = true, bool freezen = false)
				: SunshineBaseLayer(LayerType::Conv, learnRate, user_alias, activate, freezen)
			{
				if (shape.size() < 2)throw std::runtime_error("ConvLayer2D construct err : shape less than 2...");
				//int add_size = 3 - shape.size();
				// std::list<int>tmpShape = shape;
				//for (int i = 0; i < add_size; ++i) {
				//	tmpShape(1);
				//}
				m_filterSize = filterSize;
				m_padding = padding;
				m_step = step;
				m_weightMat = Algebra::CMatrix(shape);
				m_biasMat = Algebra::CMatrix::ones({shape.front(), 1 });//ÿһ�������ľ����ӵ��һ��bias
				m_weightMat.random_normalize(0, 1);
				m_biasMat.random_normalize(0, 1);
				/*
					m_backUpdateMat = m_weightMat;
					m_backUpdateBiasMat = Algebra::CMatrix::ones({ shape.back(), 1 }) * m_layerLearningRate;
					m_biasMat = m_backUpdateBiasMat;
					//��ʼ���������
					m_weightMat.random_normalize(0, 1);
					m_biasMat.random_normalize(0, 1);
				*/
				
			};
			~ConvLayer2D() {};
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
					*(m_weightMat.getdataptr().get() + i) = std::strtod(data.c_str(), nullptr);
				}
				for (int i = 0; i < m_biasMat.gettotalsize(); ++i) {
					std::string&& data = findValue(para, SPLITCHAR);
					Algebra::MatrixDataType a = std::strtod(data.c_str(), nullptr);
					*(m_biasMat.getdataptr().get() + i) = std::strtod(data.c_str(), nullptr);
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
			void calBackUpdateMat(const Algebra::CMatrix& loss) override
			{
				return;
			};
			void updateBackMat(const Algebra::CMatrix& loss) override
			{
				return;
			};
			void showWeight() override {
				SunshineBaseLayer::showWeight();
				m_weightMat.print();
				m_biasMat.print();
			}

			void forwardMove(const Algebra::CMatrix& feedIn) override
			{
				auto expandMat = matrixExpand(feedIn);
				//william resume
		/*		auto prevShape = m_weightMat.shape();
				m_weightMat.reshape({1, m_weightMat.gettotalsize()});
				auto tmpOutMat = Algebra::CMatrix::matmul(m_weightMat, expandMat);
				m_weightMat.reshape(prevShape);
				auto outMat = geneOutShape(feedIn.shape(), m_step);
				tmpOutMat.reshape(outMat);
				m_front2backMat = tmpOutMat;*/
				//end william
				return;
			}
			void forwardMove() override
			{
				return;
			}
			void backwardMove() override
			{
				return;
			}

			public:	
			/*
			���������ƽ�̣����㽫�������ת��Ϊ����˷�����
			*/
			Algebra::CMatrix matrixExpand(const Algebra::CMatrix& feedIn) {
				auto feedInShape = feedIn.shape();
				std::vector<int> vecfeedInShape{ feedInShape.rbegin(),feedInShape.rend() };
				auto outShape = geneOutShape(feedIn.shape(), m_step);
				//william resume
				std::vector<int> vecOutShape{ outShape.rbegin(), outShape.rend() };//�ų���������Ϣ,�����������*��*��
				auto tmpWeightMat = m_weightMat.shape();
				std::vector<int>kernelShape{ tmpWeightMat.rbegin(), tmpWeightMat.rend() };
				int expandMatShapeCol = 1;
				int expandMatShapeRow = 1;
				for (auto& i : kernelShape) {
					expandMatShapeCol *= i;
				}
				for (auto& i : vecOutShape) {
					expandMatShapeRow *= i;
				}
				Algebra::CMatrix retMat({expandMatShapeRow, expandMatShapeCol});
				auto retMatDataPtr = retMat.getdataptr();
				auto feedInMatDataPtr = feedIn.getdataptr();
				//���retMat
				std::vector<Algebra::MatrixDataType> cacheData(expandMatShapeCol, 0);
				cacheData.reserve(expandMatShapeCol);
				int feedInFloorSize = vecfeedInShape[1] * vecfeedInShape[0];
				int cacheDataIdx = 0;
				int nowOutFeedInIdx = 0;
				bool jumpOut = false;
				bool edgeAlarm = false;
				bool cacheInitOk = false;
				for (int feedInRow = 0; feedInRow < vecfeedInShape[1]; feedInRow += m_step.first) {
					if (jumpOut)break;
					edgeAlarm = false;
					for (int feedInCol = 0; feedInCol < vecfeedInShape[0]; feedInCol += m_step.second) {
						cacheDataIdx = 0;
						for (int kd = 0; kd < kernelShape[2]; ++kd) {
							for (int kr = 0; kr < kernelShape[1]; ++kr) {
								//�߽���
								if (feedInRow + kr >= vecfeedInShape[1]) {
									jumpOut = true;
									edgeAlarm = true;
									break;
								}
								for (int kc = 0; kc < kernelShape[0]; ++kc) {
									//�߽���
									if (feedInCol + kc >= vecfeedInShape[0]) {
										edgeAlarm = true;
										break;
									}
									const auto& data = feedInMatDataPtr[kd * feedInFloorSize + (feedInRow + kr) * vecfeedInShape[0] + (feedInCol + kc)];
									cacheData[cacheDataIdx++] = data;

								}
								if (edgeAlarm)break;
							}
							if (edgeAlarm)break;
						}
						if (edgeAlarm)break;
						//�߽���ͨ��
						if (cacheData.size() != expandMatShapeCol)throw std::runtime_error("cacheData.size() != expandMatShapeRow-- " + m_usrAlias);
						for (const auto& i : cacheData) {
							retMatDataPtr[nowOutFeedInIdx++] = i;
						}
					}
				}
				return retMat;
			}

			/*
			*�������������������ߴ�
			*Ŀǰʹ�õ��ǲ��Եķ�ʽ�����������ĳߴ�
			*feedIn:�������Ҫ���о���ľ���
			*step:ÿ��ƫ�ƵĲ��� <�в���,�в���>
			*/
			 std::list<int>geneOutShape(const std::list<int> feedIn, const std::pair<int,int>step) {
				judgeValid(feedIn);
				std::vector<int> feedInShape(feedIn.rbegin(), feedIn.rend());
				auto tmpWeighMat = m_weightMat.shape();
				std::vector<int> weightShape(tmpWeighMat.rbegin(), tmpWeighMat.rend());
				// �˴�Ϊpadding֮������ݣ����� pΪ0
				int col = (feedInShape[0] - weightShape[0]) / step.second + 1;
				int row = (feedInShape[1] - weightShape[1]) / step.first + 1;
				return {row, col };
				
				//�Դ�棬���õ���
				//std::list<int> feedInShape(feedIn.rbegin(), feedIn.rend());
				//auto tmpWeighMat = m_weightMat.shape();
				//std::list<int> weightShape(tmpWeighMat.rbegin(), tmpWeighMat.rend());
				//int final_row = -1, final_col = -1;
				////try ��
				//for (int col = 0, count=0; col < feedInShape[0];++count) {
				//	if (final_col != -1)break;
				//	for (int weightCol = 0; weightCol < weightShape[0]; ++weightCol) {
				//		if (col + weightCol >= feedInShape[0]) {
				//			final_col = count;
				//			break;
				//		}
				//	}
				//	col += step.second;
				//}
				////try ��
				//for (int row = 0 , count = 0; row < feedInShape[1];++count) {
				//	if (final_row != -1)break;
				//	for (int weightRow = 0; weightRow < weightShape[1]; ++weightRow) {
				//		if (row + weightRow >= feedInShape[1]) {
				//			final_row = count;
				//			break;
				//		}
				//	}
				//	row += step.first;
				//}
				//if (final_row == -1 || final_col == -1)throw std::runtime_error("geneOutShape shutdown: Something wrong...final_row or final_col equal 1--" + m_usrAlias);
				//return { m_weightMat.shape().front() ,final_row, final_col };
			}



			private:
			/*�жϴ�������͸þ���Ƿ���������������
			����ͼƬ����Ӧ�ô��ھ���˵����У�����ͼƬ���Ӧ��������һ��
			feedIn:����ĳߴ��С 
			*/
			void judgeValid(const std::list<int>& feedIn){
				auto tmpFeedIn = feedIn;
				if (tmpFeedIn.size() < 2) throw std::runtime_error("Conv: judgeValid ----dim not less than 2") ;
				std::vector<int> checkShape{ tmpFeedIn.rbegin(), tmpFeedIn.rend() };
				 std::list<int>tmpWeightShape = m_weightMat.shape();
				std::vector<int> weightShape{ tmpWeightShape.rbegin(), tmpWeightShape.rend() };
				int weightDim = weightShape.size();
				for (int i = 0; i < weightDim; ++i) {
					if (i == 2) {
						if (weightShape[i] != checkShape[i])throw std::runtime_error("Conv: judgeValid --weightshape dim 2 don't equal checkshape dim 2.." + m_usrAlias);
					}
					else if (i < 2) {
						if (weightShape[i] > checkShape[i])throw std::runtime_error("Conv: judgeValid --weightshape can't bigger than checkshape.." + m_usrAlias);
					}
					else break;
				}
			}

			Algebra::CMatrix Padding(const Algebra::CMatrix& feedIn) {

			}
			int m_filterSize;
			std::string m_padding;
			std::pair<int, int>m_step;
			Algebra::CMatrix m_weightMat;
			Algebra::CMatrix m_biasMat;	      
			Algebra::CMatrix m_backUpdateMat;    
			Algebra::CMatrix m_backUpdateBiasMat; 
		};

	}

}