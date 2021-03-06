#include <iostream>
#include <memory>
#include <list>
#include <thread>
#ifdef _MSC_VER
	#include <filesystem>
#else
#endif
#include "SunshineNet.hpp"
#include "gtest/gtest.h"


TEST(CMATRIX, broadCast) {
	SunshineFrame::Algebra::CMatrix a({2,2,5,3});
	SunshineFrame::Algebra::CMatrix b({ 3,1,2,1,3 });
	 std::list<int> out_shape;
	 std::list<int >b_tmp = { 3,2,2,5,3 };
	EXPECT_TRUE(SunshineFrame::Algebra::CMatrix::broadcastRule(a.shape(), b.shape(), out_shape));
	EXPECT_TRUE(std::equal(out_shape.begin(), out_shape.end(), b_tmp.begin(),b_tmp.end()));
}

TEST(CMATRIX, transposition) {
	SunshineFrame::Algebra::CMatrix a({2,2});
	a.matrixFeed({ 1,2,3,4 });
	auto b = a.T();
	SunshineFrame::Algebra::CMatrix c({ 2,2 });
	c.matrixFeed({ 1,3,2,4 });
	EXPECT_EQ(b ,c );

	SunshineFrame::Algebra::CMatrix a1({6,1});
	a1.matrixFeed({ 1,2,3,4,5,6 });
	auto b1 = a1.T();
	SunshineFrame::Algebra::CMatrix c1({ 1,6 });
	c1.matrixFeed({1,2,3,4,5,6});
	EXPECT_EQ(b1 ,c1);

	SunshineFrame::Algebra::CMatrix a2({ 3,1,2 });
	a2.matrixFeed({ 1,2,3,4,5,6 });
	auto b2 = a2.T();
	SunshineFrame::Algebra::CMatrix c2({ 2,1,3 });
	c2.matrixFeed({1,3,5,2,4,6});
	EXPECT_EQ(b2, c2);

	SunshineFrame::Algebra::CMatrix a3({ 3,2,2 });
	a3.matrixFeed({ 1,2,3,4,5,6,7,8,9,10,11,12 });
	auto b3 = a3.T();
	SunshineFrame::Algebra::CMatrix c3({ 2,2,3 });
	c3.matrixFeed({1,5,9,3,7,11,2,6,10,4,8,12});
	EXPECT_EQ(b3, c3);

	SunshineFrame::Algebra::CMatrix a4({ 1,3,2,2 });
	a4.matrixFeed({ 1,2,3,4,5,6,7,8,9,10,11,12 });
	auto b4 = a4.T();
	SunshineFrame::Algebra::CMatrix c4({ 2,2,3,1});
	c4.matrixFeed({ 1,5,9,3,7,11,2,6,10,4,8,12 });
	EXPECT_EQ(b4, c4);
}

TEST(CMATRIX, matmul) {
	//??ά?˷?
	//SunshineFrame::Algebra::CMatrix a({5,4 });
	//a.matrixFeed({ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 });
	//SunshineFrame::Algebra::CMatrix b({ 4,3,4,1 });
	//for (int i = 1; i <= 48; ++i) {
	//	b.getdataptr()[i - 1] = i;
	//}
	//auto c = SunshineFrame::Algebra::CMatrix::matmul(a, b);
	//c.print();
	//??ͨ?˷?
	SunshineFrame::Algebra::CMatrix a({5,4 });
	a.matrixFeed({ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 });
	SunshineFrame::Algebra::CMatrix b({ 4,1 });
	for (int i = 1; i <= 4; ++i) {
		b.getdataptr()[i - 1] = i;
	}
	auto c = SunshineFrame::Algebra::CMatrix::matmul(a, b);
	SunshineFrame::Algebra::CMatrix right({5,1});
	right.matrixFeed({ 30,70,110,150,190 });
	EXPECT_EQ(c, right);
}

//save para test
TEST(LAYERTEST, fc)
{
	auto fc = []()->bool {
			try {
			SunshineFrame::Sunshine frame;
			SunshineFrame::Layer::FullyConnectLayer f1({ 4,3 },  0.00005, "Huyue");
			SunshineFrame::Layer::FullyConnectLayer f3({ 1,4 },  0.00005, "linan");
			//f1.weightMatFeed({ 3, 5 });
			//f1.biasMatFeed({ 1 });
			auto f1_out = frame.addFullyConnectedLayer(f1);
			auto f2_out = frame.AddReluLayer({ f1_out }, "relu");
			auto f3_out = frame.addFullyConnectedLayer({ f2_out }, f3);
			int id = frame.AddLossLayer({ f3_out }, SunshineFrame::Layer::LossLayerType::MSE, "mse");
			int count = 0;
			while (true)
			{
				SunshineFrame::Algebra::CMatrix feedData({ 4, 3 });
				feedData.matrixFeed(std::list<SunshineFrame::Algebra::MatrixDataType>{ 1,2,1,2,-1,3,5,-2,2,-5,2,-3 });
				feedData = feedData.T();
				SunshineFrame::Algebra::CMatrix target({ 1,4 });
				target.matrixFeed(std::list<SunshineFrame::Algebra::MatrixDataType>{ 27,40,401,-176 });
				std::map<int, SunshineFrame::Algebra::CMatrix> mapTar;
				mapTar[id] = target;
				frame.train(feedData, mapTar);
				//frame.frameShowAllData();
				using namespace std::chrono_literals;
				//std::this_thread::sleep_for(0.01s);
				if (count >= 800)break;
				count++;
			}
			SunshineFrame::Algebra::CMatrix right{ 1,1 };
			right.matrixFeed({ 0.001 });
			auto ww = frame.getLossLayer(id)->m_front2backMat;
			EXPECT_LE( frame.getLossLayer(id)->m_front2backMat, right);

			SunshineFrame::Algebra::CMatrix testData({ 3,1 });
			testData.matrixFeed(std::list<SunshineFrame::Algebra::MatrixDataType>{ 5,1,2 });
			frame.predict(testData);
			//frame.frameShowAllData();
			auto save_result = f3_out->getFront2BackMat();
			frame.save(std::filesystem::current_path() / "weight", "william.txt");

			SunshineFrame::Sunshine frame2;
			SunshineFrame::Layer::FullyConnectLayer f11({ 4,3 }, 0.00005);
			SunshineFrame::Layer::FullyConnectLayer f31({ 1,4 }, 0.00005);
			//f1.weightMatFeed({ 3, 5 });
			//f1.biasMatFeed({ 1 });
			auto f1_out1 = frame2.addFullyConnectedLayer(f11);
			auto f2_out1 = frame2.AddReluLayer({ f1_out1 }, "relu");
			auto f3_out1 = frame2.addFullyConnectedLayer({ f2_out1 }, f31);
			frame2.AddLossLayer({ f3_out1 }, SunshineFrame::Layer::LossLayerType::MSE ,"mse");
			frame2.load(std::filesystem::current_path() / "weight/william.txt");
			//std::cout << "predict....after\n";
			SunshineFrame::Algebra::CMatrix testData1({ 3,1 });
			testData1.matrixFeed(std::list<SunshineFrame::Algebra::MatrixDataType>{ 5,1,2 });
			frame2.predict(testData1);
			auto loadResult = f3_out1->getFront2BackMat();
			//frame2.frameShowAllData();
			if (save_result == loadResult)return true;
			else return false;
		}
		catch (std::runtime_error e) {
			std::cout << e.what() << std::endl;
		}
		return false;
	};
	EXPECT_TRUE(fc());
}
//



TEST(LAYERTEST, conv_geneOutShape) {
	SunshineFrame::Layer::ConvLayer2D a(2 ,{3,3,3});
	auto out = a.geneOutShape({1,3,7,7 }, {2,2});
	 std::list<int> right{1,2, 3,3 };
	EXPECT_TRUE(std::equal(out.begin(), out.end(), right.begin(),right.end()));
	SunshineFrame::Layer::ConvLayer2D b(1, {5,3,3});
	out = b.geneOutShape({ 1,5,10,10 }, { 2,1 });
	right = {1,1, 4,8 };
	EXPECT_TRUE(std::equal(out.begin(), out.end(), right.begin(), right.end()));

	//

}
TEST(LAYERTEST, conv_matrixExpand) {
	SunshineFrame::Layer::ConvLayer2D a(1, {2,2,3 }, "noPadding", {2,1});
	SunshineFrame::Algebra::CMatrix feedIn({1,2,4,5});
	 std::list<SunshineFrame::Algebra::MatrixDataType>feedInlist;
	for (int i = 1; i <= 40; ++i) {
		feedInlist.push_back(i);
	}
	feedIn.matrixFeed(feedInlist);
	auto expand = a.matrixExpand(feedIn);
	//expand.print();
	SunshineFrame::Algebra::CMatrix right({1, 1, 6, 12});
	right.matrixFeed(std::list<SunshineFrame::Algebra::MatrixDataType>{1, 2, 3, 6, 7, 8, 21, 22, 23, 26, 27, 28, 2, 3, 4, 7, 8, 9, 22, 23, 24, 27, 28, 29, 3, 4, 5, 8, 9, 10, 23, 24, 25, 28, 29, 30, 11, 12, 13, 16, 17, 18, 31, 32, 33, 36, 37, 38, 12, 13, 14, 17, 18, 19, 32, 33, 34, 37, 38, 39, 13, 14, 15, 18, 19, 20, 33, 34, 35, 38, 39, 40});
	EXPECT_EQ(expand, right);
}


TEST(LAYERTEST, conv_forwardTest) {
	//[[[[710,818]]]]
	SunshineFrame::Sunshine frame;
	SunshineFrame::Layer::ConvLayer2D a(1, {2,2,2 }, "noPadding", { 1,1 });
	a.weightMatFeed({ 10,11,12,13,14,15,16,17 });
	SunshineFrame::Algebra::CMatrix feedIn({ 1,2, 2, 3 });
	feedIn.matrixFeed({ 1,2,3,4,5,6,7,8,9,10,11,12 });
	auto a_itr = frame.addConv2DLayer(a);
	frame.predict(feedIn);
	auto data = a_itr->getFront2BackMat();
	SunshineFrame::Algebra::CMatrix right{1,1,1,2};
	right.matrixFeed({710,818});
	EXPECT_EQ(data, right);
	//SunshineFrame::Sunshine frame;
	//SunshineFrame::Layer::ConvLayer2D a(1, { 3,5,4 }, "noPadding", { 1,1 });
	//SunshineFrame::Algebra::CMatrix feedIn({1,3,1024,768 });
	//auto a_itr = frame.addConv2DLayer(a);
	//frame.predict(feedIn);
}

