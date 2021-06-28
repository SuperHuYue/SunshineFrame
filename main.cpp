#include <iostream>
#include <memory>
#include <list>
#include <thread>
#include <filesystem>
#include "SunshineNet.hpp"
#include "gtest/gtest.h"



//save para test
TEST(LAYERTEST, fc)
{
	auto fc = []()->bool {
			try {
			SunshineFrame::Sunshine frame;
			SunshineFrame::Layer::FullyConnectLayer f1({ 4,3 }, 0.00005, "Huyue");
			SunshineFrame::Layer::FullyConnectLayer f3({ 1,4 }, 0.00005, "linan");
			//f1.weightMatFeed({ 3, 5 });
			//f1.biasMatFeed({ 1 });
			auto f1_out = frame.addFullyConnectedLayer(f1);
			auto f2_out = frame.AddReluLayer({ f1_out });
			auto f3_out = frame.addFullyConnectedLayer({ f2_out }, f3);
			int id = frame.AddLossLayer({ f3_out });
			int count = 0;
			while (true)
			{
				SunshineFrame::Algebra::CMatrix feedData({ 4, 3 });
				feedData.matrixFeed({ 1,2,1,2,-1,3,5,-2,2,-5,2,-3 });
				feedData = feedData.T();
				SunshineFrame::Algebra::CMatrix target({ 1,4 });
				target.matrixFeed({ 27,40,401,-176 });
				std::map<int, SunshineFrame::Algebra::CMatrix> mapTar;
				mapTar[id] = target;
				frame.train(feedData, mapTar);
				using namespace std::chrono_literals;
				std::this_thread::sleep_for(0.01s);
				count++;
				//std::cout << "count = " << count << "\n";
				if (count == 50)break;
			}
			//std::cout << "predict....\n";
			SunshineFrame::Algebra::CMatrix testData({ 3,1 });
			testData.matrixFeed({ 5,1,2 });
			frame.predict(testData);
			auto save_result = f3_out->getFront2BackMat();
			frame.save(std::filesystem::current_path() / "weight", "william.txt");

			SunshineFrame::Sunshine frame2;
			SunshineFrame::Layer::FullyConnectLayer f11({ 4,3 }, 0.00005);
			SunshineFrame::Layer::FullyConnectLayer f31({ 1,4 }, 0.00005);
			//f1.weightMatFeed({ 3, 5 });
			//f1.biasMatFeed({ 1 });
			auto f1_out1 = frame2.addFullyConnectedLayer(f11);
			auto f2_out1 = frame2.AddReluLayer({ f1_out1 });
			auto f3_out1 = frame2.addFullyConnectedLayer({ f2_out1 }, f31);
			frame2.AddLossLayer({ f3_out1 });

			frame2.load(std::filesystem::current_path() / "weight/william.txt");
			//std::cout << "predict....after\n";
			SunshineFrame::Algebra::CMatrix testData1({ 3,1 });
			testData1.matrixFeed({ 5,1,2 });
			frame2.predict(testData1);
			auto loadResult = f3_out1->getFront2BackMat();
			if (save_result == loadResult)return true;
			else return false;
			//frame2.frameShowAllData();
		}
		catch (std::runtime_error e) {
			std::cout << e.what() << std::endl;
		}
		return false;
	};
	EXPECT_TRUE(fc());
}

