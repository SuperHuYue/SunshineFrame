#pragma once
#include "baselayer.hpp"
#include <array>
/*
卷积层目前尚未完成，赶工加工中
*/
namespace SunshineFrame {
	namespace Layer {
		class ConvLayer2D: public SunshineBaseLayer
		{
			const char SPLITCHAR = ',';
		public:
			ConvLayer2D() = delete;
			///*
			//	卷积层:
			//	filterSize:核的个数
			//	shape: batchsize,深，行，列 （与CMatrix一致）
			//	padding: no--- nopadding, same---尚未实现,
			//	step:行移动步长，列移动步长
			//*/
			//ConvLayer2D(const int filterSize, const std::array<int,4>& shape, const std::string padding = "noPadding", const std::pair<int, int>& step = { 1,1 }, const double& learnRate = 0.005, const std::string& user_alias = "undefined", bool activate = true, bool freezen = false)
			//	: SunshineBaseLayer(LayerType::Conv, learnRate, user_alias, activate, freezen)
			//{
			//	if (Size != 4)throw std::runtime_error("ConvLayer2D size must equal 4..");
			//	m_filterSize = filterSize;
			//	m_padding = padding;
			//	m_step = step;
			//	m_kernelShape[2] = shape[3];
			//	m_kernelShape[1] = shape[2];
			//	m_kernelShape[0] = shape[1];
			//	m_weightMat = Algebra::CMatrix({ m_filterSize,shape[1], shape[2],shape[3] });
			//	//m_biasMat = Algebra::CMatrix::ones({shape.front(), 1 });//需要加入此部分的考虑(尚未完成2021/07/01)
			//	m_weightMat.random_normalize(0, 1);
			//	m_biasMat.random_normalize(0, 1);
			//	/*
			//		m_backUpdateMat = m_weightMat;
			//		m_backUpdateBiasMat = Algebra::CMatrix::ones({ shape.back(), 1 }) * m_layerLearningRate;
			//		m_biasMat = m_backUpdateBiasMat;
			//		//初始化随机种子
			//		m_weightMat.random_normalize(0, 1);
			//		m_biasMat.random_normalize(0, 1);
			//	*/
			//	
			//};
			ConvLayer2D(const int filterSize, const std::initializer_list<int>shape , const std::string padding = "noPadding", const std::pair<int, int>& step = { 1,1 }, const double& learnRate = 0.005, const std::string& user_alias = "undefined", bool activate = true, bool freezen = false)
				: SunshineBaseLayer(LayerType::Conv, learnRate, user_alias, activate, freezen)
			{
				if (shape.size() != 3)throw std::runtime_error("ConvLayer2D size must equal 3..");
				std::array<int, 3>arr;
				int count = 0;
				for (auto& i : shape) {
					arr[count++] = i;
				}
				m_filterSize = filterSize;
				m_padding = padding;
				m_step = step;
				m_kernelShape[2] =arr[2];//col
				m_kernelShape[1] =arr[1];//row
				m_kernelShape[0] =arr[0];//deepth
				m_weightMat = Algebra::CMatrix({ m_filterSize, arr[0], arr[1],arr[2] });
				//m_biasMat = Algebra::CMatrix::ones({shape.front(), 1 });//需要加入此部分的考虑(尚未完成2021/07/01)
				m_weightMat.random_normalize(0, 1);
				m_biasMat.random_normalize(0, 1);
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
				//auto conv_result = convInner(feedIn);
				//m_front2backMat = conv_result;

				auto conv_result = convInner(feedIn);
				conv_result.reshape(geneOutShape(feedIn.shape(), m_step));
				m_front2backMat = conv_result;
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
			获得卷积的结果
			*/
			Algebra::CMatrix convInner(const Algebra::CMatrix& feedIn) {
				if (feedIn.shape().size() != 4)throw std::runtime_error("feedIn data must dim 4..");
				if( m_kernelShape.size() != 3)throw std::runtime_error("kernel shape must dim 3..");
				Algebra::CMatrix matExpand = matrixExpand(feedIn);
				Algebra::CMatrix tmpWeight = m_weightMat;
				auto vecWeightShape = tmpWeight.vecShape();
				tmpWeight.reshape({ 1, m_filterSize, vecWeightShape[3] * vecWeightShape[1] * vecWeightShape[2] ,1 });
				auto out =  Algebra::CMatrix::matmul(matExpand, tmpWeight);
				return std::move(out);
				//auto feedInShape = feedIn.shape();
				//std::vector<int> vecfeedInShape{ feedInShape.begin(),feedInShape.end() };
				//auto outShape = geneOutShape(feedIn.shape(), m_step);
				//Algebra::CMatrix convResult({ outShape });
				//auto convResultPtr = convResult.getdataptr();
				//std::vector<int> vecOutShape{ outShape.begin(), outShape.end() };//排除掉个数信息,仅仅保留深度*行*列
				//auto weightPtr = m_weightMat.getdataptr();
				////auto tmpWeightMat = m_weightMat.shape();
				////std::vector<int>kernelShape{ tmpWeightMat.crbegin(), tmpWeightMat.crend() };

				////std::array<int,3>kernelShape{ m_kernelShape.crbegin(), m_kernelShape.crend() };
				///*int expandMatShapeCol = 1;
				//int expandMatShapeRow = 1;
				//for (auto& i : m_kernelShape) {
				//	expandMatShapeCol *= i;
				//}
				//for (auto& i : vecOutShape) {
				//	expandMatShapeRow *= i;
				//}*/
				//auto feedInMatDataPtr = feedIn.getdataptr();
				////填充retMat
				//int feedInOneStageSize = vecfeedInShape[1] * vecfeedInShape[2];
				////int cacheDataIdx = 0;
				//int nowOutFeedInIdx = 0;
				//int sinCovResult = 0;
				//int kernelOneStageSize = m_kernelShape[2] * m_kernelShape[1];
				//int prevFeedInRow = 0, prevFeedInCol = 0;
				//int prevKd = 0, kr = 0, kc = 0;
				//int kdmulFeedInOneStageSize = 0;
				//int kdMulKernelOneStageSize = 0;
				//int krMulKernelShapeTwo = 0;
				//int krMulFeedInShapeTwo = 0;
				//int feedInRowMulFeedInShapeTwo = 0;
				//int feedInRowMulOutShapeOne = 0;
				//for (int feedInRow = 0; feedInRow < vecfeedInShape[1] - m_kernelShape[1] + 1; feedInRow += m_step.first) {
				//	feedInRowMulFeedInShapeTwo = feedInRow * vecfeedInShape[2];
				//	feedInRowMulOutShapeOne = feedInRow * vecOutShape[1];
				//	for (int feedInCol = 0; feedInCol < vecfeedInShape[2] - m_kernelShape[2] + 1; feedInCol += m_step.second) {
				//		memset(&sinCovResult, 0, sizeof(int));
				//		for (int kd = 0; kd < m_kernelShape[0]; ++kd) {
				//			kdmulFeedInOneStageSize = kd * feedInOneStageSize;
				//			kdMulKernelOneStageSize = kd * kernelOneStageSize;
				//			for (int kr = 0; kr < m_kernelShape[1]; ++kr) {
				//				krMulKernelShapeTwo = kr * m_kernelShape[2];
				//				krMulFeedInShapeTwo = kr * vecfeedInShape[2];
				//				for (int kc = 0; kc < m_kernelShape[2]; ++kc) {
				//					 //const auto& data = feedInMatDataPtr[size_t(kd * feedInOneStageSize + (feedInRow + kr) * vecfeedInShape[0] + (feedInCol + kc))];
				//					 sinCovResult += weightPtr[kc + krMulKernelShapeTwo + kdMulKernelOneStageSize] *  feedInMatDataPtr[size_t(kdmulFeedInOneStageSize + feedInRowMulFeedInShapeTwo+ krMulFeedInShapeTwo + (feedInCol + kc))];
				//					 int a = weightPtr[kc + krMulKernelShapeTwo + kdMulKernelOneStageSize];
				//					 int b = feedInMatDataPtr[size_t(kdmulFeedInOneStageSize + feedInRowMulFeedInShapeTwo+ krMulFeedInShapeTwo + (feedInCol + kc))];
				//					 std::cout << "-----------" << a <<"---"<<b<< "\n";
				//				}
				//			}
				//		}
				//		convResultPtr[feedInRowMulOutShapeOne  + feedInCol] = sinCovResult;
				//	}
				//}
				//return std::move(convResult);
			}


			//			
			//将矩阵进行平铺，方便将卷积运算转变为矩阵乘法运算 cache
			//
			Algebra::CMatrix matrixExpand(const Algebra::CMatrix& feedIn) {
				auto feedInShape = feedIn.shape();
				std::vector<int>vecFeedInShape{ feedInShape.begin(), feedInShape.end() };
				auto outShape = geneOutShape(feedIn.shape(), m_step);
				std::vector<int>vecOutShape{ outShape.begin(), outShape.end() };
				std::array<int, 3>arrExpandShape;
				int expandMatShapeCol = 1;
				int expandMatShapeRow = vecOutShape[2] * vecOutShape[3];
				std::vector<int>vecKernelShape{ m_kernelShape.begin(), m_kernelShape.end() };
				for (auto& i : m_kernelShape) {
					expandMatShapeCol *= i;
				}
				arrExpandShape.fill(0);
				arrExpandShape[0] = vecFeedInShape[0];//batch
				arrExpandShape[1] = expandMatShapeRow;//row
				arrExpandShape[2] = expandMatShapeCol;//col
				Algebra::CMatrix expandMat({ arrExpandShape[0],1 , arrExpandShape[1],arrExpandShape[2] });
				auto expandMatDataPtr = expandMat.getdataptr();
				auto feedInMatDataPtr = feedIn.getdataptr();
				int nowExpandIdx = 0;
				int feedInFloorSize = vecFeedInShape[2] * vecFeedInShape[3];//row * col
				int feedInStageSize = feedInFloorSize * vecFeedInShape[1];//deepth*row*col
				int batchOffset = 0;
				int deepthOffset = 0;
				int feedInRowOffset = 0;
				int kernelRowOffset = 0;
				for (int batchIdx = 0; batchIdx < vecFeedInShape[0]; ++batchIdx){
					batchOffset = batchIdx * feedInStageSize;
					for (int feedInRow = 0; feedInRow < vecFeedInShape[2] - vecKernelShape[1] + 1; feedInRow += m_step.first) {
						feedInRowOffset = feedInRow * vecFeedInShape[3];
						for (int feedInCol = 0; feedInCol < vecFeedInShape[3] - vecKernelShape[2] + 1; feedInCol += m_step.second) {
							for (int kd = 0; kd < vecKernelShape[0]; ++kd) {
								deepthOffset = kd * feedInFloorSize;
								for (int kr = 0; kr < vecKernelShape[1]; ++kr) {
									kernelRowOffset = kr * vecFeedInShape[3];
									for (int kc = 0; kc < vecKernelShape[2]; ++kc) {
										//expandMatDataPtr[nowExpandIdx++] = feedInMatDataPtr[batchOffset + kd * feedInFloorSize + (feedInRow + kr) *vecFeedInShape[3]+ feedInCol + kc];
										expandMatDataPtr[nowExpandIdx++] = feedInMatDataPtr[batchOffset + deepthOffset + feedInRowOffset + kernelRowOffset + feedInCol + kc];
									}
								}
							}
						}
					}
				}
				return std::move(expandMat);
				//Algebra::CMatrix retMat({vecfeedInShape[3],expandMatShapeRow, expandMatShapeCol });
				//auto retMatDataPtr = retMat.getdataptr();
				//auto feedInMatDataPtr = feedIn.getdataptr();
				////填充retMat
				//int feedInFloorSize = vecfeedInShape[1] * vecfeedInShape[0];
				//int feedInStageSize = feedInFloorSize * vecfeedInShape[2];
				//int nowOutFeedInIdx = 0;
				//for (int feedInBatch = 0; feedInBatch < vecfeedInShape[3]; ++feedInBatch) {
				//	for (int feedInRow = 0; feedInRow < vecfeedInShape[1] - kernelShape[1] + 1; feedInRow += m_step.first) {
				//		for (int feedInCol = 0; feedInCol < vecfeedInShape[0] - kernelShape[0] + 1; feedInCol += m_step.second) {
				//			for (int kd = 0; kd < kernelShape[2]; ++kd) {
				//				for (int kr = 0; kr < kernelShape[1]; ++kr) {
				//					for (int kc = 0; kc < kernelShape[0]; ++kc) {
				//						const auto& data = feedInMatDataPtr[size_t(kd * feedInFloorSize + (feedInRow + kr) * vecfeedInShape[0] + (feedInCol + kc) + feedInBatch * feedInStageSize)];
				//						retMatDataPtr[nowOutFeedInIdx++] = data;

				//					}
				//				}
				//			}
				//		}
				//	}
				//}
				//return std::move(retMat);
			}

			/*
			*根据输入生成输出对象尺寸
			*feedIn:传入的需要进行卷积的矩阵
			*step:每次偏移的步长 <行步长,列步长>
			*/
			 std::list<int>geneOutShape(const std::list<int> feedIn, const std::pair<int,int>step) {
				judgeValid(feedIn);
				std::vector<int> feedInShape(feedIn.begin(), feedIn.end());
				// 此处为padding之后的内容，所以 p为0
				int col = (feedInShape[feedInShape.size() - 1] - m_kernelShape[m_kernelShape.size() - 1]) / step.second + 1;
				int row = (feedInShape[feedInShape.size() - 2] - m_kernelShape[m_kernelShape.size() - 2]) / step.first + 1;
				return {feedInShape[0],m_filterSize,row, col };//batch,filtersize,row,col
				//试错版，可用但慢
				//std::list<int> feedInShape(feedIn.rbegin(), feedIn.rend());
				//auto tmpWeighMat = m_weightMat.shape();
				//std::list<int> weightShape(tmpWeighMat.rbegin(), tmpWeighMat.rend());
				//int final_row = -1, final_col = -1;
				////try 列
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
				////try 行
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
			/*判断传入参数和该卷积是否满足卷积运算条件
			输入图片行列应该大于卷积核的行列，输入图片深度应该与核深度一致
			feedIn:输入的尺寸大小 
			*/
			void judgeValid(const std::list<int>& feedIn){
				//feedin [batch,deepth,row,col] 0->4
				//kernel [deepth,row,col] 0->3
				auto tmpFeedIn = feedIn;
				if (tmpFeedIn.size() != 4) throw std::runtime_error("Conv: judgeValid ----dim must equal 4") ;
				std::vector<int> checkShape{ tmpFeedIn.begin(), tmpFeedIn.end()};
				if(m_kernelShape[0] != checkShape[1])throw std::runtime_error("Conv: judgeValid --deep not equal.." + m_usrAlias);
				if(m_kernelShape[1] > checkShape[2] || m_kernelShape[2] > checkShape[3])throw std::runtime_error("Conv: judgeValid----kernel bigger than checkshape" + m_usrAlias);
			}

			Algebra::CMatrix Padding(const Algebra::CMatrix& feedIn) {

			}
			int m_filterSize;
			//int m_batchSize;
			std::array<int, 3> m_kernelShape;
			std::string m_padding;
			std::pair<int, int>m_step;
			Algebra::CMatrix m_weightMat;//shape = [m_fileterSize, m_kernelShape];
			Algebra::CMatrix m_biasMat;	      
			Algebra::CMatrix m_backUpdateMat;    
			Algebra::CMatrix m_backUpdateBiasMat; 
		};

	}

}