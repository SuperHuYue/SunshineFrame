#pragma once
#include "CMatrix.hpp"
#include <iostream>
#include <list>
#include <string>
#include <sstream>
#include <filesystem>
#include <fstream>
#include "alllayers.hpp"
namespace SunshineFrame {

	class Sunshine {
	public:
		Sunshine():m_bInit(true),m_lossIdValid(0){};
		~Sunshine() {
		}
		void frameShowAllData() {
			for (auto i : m_layer) {
				for (auto j : i.second) {
					//std::cout << "alisa:  " << j->alias2String() <<"\n";
					j->showWeight();
					j->getFront2BackMat().print();
					//j->getBack2FrontMat().print();
				}
				std::cout << "---------------------------------\n";
			}
		}
		/*
		����ѵ��Ȩֵ����Ҫ��鿴:_saveEncode
		*/
		void save(std::filesystem::path savePath, const std::string&fileName){
			using namespace std::filesystem;
			std::ofstream os;
			if (!exists(savePath)) {
				create_directory(savePath);
			}
			os.open(savePath/fileName, std::ios_base::out);
			auto s = status(savePath/fileName);
			if (!is_regular_file(s)) {
				throw std::runtime_error("Err not a regular file...");
			}
			_saveEncode(os);
		}
		void load(std::filesystem::path loadPath) {
			using namespace std::filesystem;
			if (!exists(loadPath)) {
				throw std::runtime_error("Err load: no such path...");
			}
			auto s = status(loadPath);
			if (!is_regular_file(s)) {
				throw std::runtime_error("Err not a regular file...");
			}
			if (m_bInit) {
				init();
				m_bInit = false;
			}
			std::ifstream ifs{ loadPath, std::ios::ate };
			int size = ifs.tellg();//һ���Զ���
			std::string cache(size, '\n');
			ifs.seekg(0);
			if (ifs.read(&cache[0], size)) {
				_loadEncode(cache);
			}
			else {
				throw std::runtime_error("Err: load file exception...");
			}

		}



		//Ϊ�������ȫ���Ӳ�
		std::shared_ptr<Layer::SunshineBaseLayer> addFullyConnectedLayer(std::list<std::shared_ptr<Layer::SunshineBaseLayer>>frontBlockIn,const Layer::FullyConnectLayer& fullyConnectIn) {
			auto fullyConnectForSharePtr = std::make_shared<Layer::FullyConnectLayer>(fullyConnectIn);
			_addLayer(frontBlockIn, fullyConnectForSharePtr);
			return fullyConnectForSharePtr;
		}
		//Ϊ�������ȫ���Ӳ�
		std::shared_ptr<Layer::SunshineBaseLayer> addFullyConnectedLayer(const Layer::FullyConnectLayer& fullyConnect) {
			auto fullyConnectForSharePtr = std::make_shared<Layer::FullyConnectLayer>(fullyConnect);
			_addLayer(fullyConnectForSharePtr);
			return fullyConnectForSharePtr;
		}


		/*Ϊ���������ʧ��
		return value:��ʧ��ָʾid�ţ�������ѵ�������и����Ӧ��id�Ľ�������
		*/
		int AddLossLayer(std::shared_ptr<Layer::SunshineBaseLayer> specifiedLayer,const Layer::LossLayerType& lossType = Layer::LossLayerType::MSE) {
			std::shared_ptr<Layer::SunshineBaseLayer> lossLayerPtr = nullptr;
			switch (lossType)
			{
			case Layer::LossLayerType::MSE:
				lossLayerPtr = std::make_shared<Layer::MseLossLayer>();
			default:
				break;
			}
			//std::lossPtr = std::make_shared<Layer::MseLossLayer>();
			if (lossLayerPtr == nullptr)throw std::runtime_error("Err: Not a Valid lossLayerType..");
			lossLayerPtr->m_iLossID = m_lossIdValid++;
			_addLayer({ specifiedLayer }, lossLayerPtr);
			return lossLayerPtr->m_iLossID;

		}
		std::shared_ptr<Layer::SunshineBaseLayer> AddReluLayer(std::shared_ptr<Layer::SunshineBaseLayer>specifiedLayer) {
			auto relu = std::make_shared<Layer::ReluLayer>();
			_addLayer({ specifiedLayer }, relu);
			return relu;
		}

		//Ԥ��׶Σ�����Ҫ��������Լ�,��ʧ�㲻���봫��
		void predict(const Algebra::CMatrix& feedIn) {
			if (m_layer.empty())throw std::runtime_error("Layer is empty..");
			if (m_bInit) {
				init();
				m_bInit = false;
			}
			auto firLayer = m_layer[0];
			for (auto firBlock : firLayer) {
				if (!firBlock->m_activation)continue;
				firBlock->forwardMove(feedIn);
			}
			//Onelayer after another
			for (auto itr = std::next(m_layer.begin()); itr != m_layer.end(); ++itr) {
				for (auto block : itr->second) {
					// m_activationΪfalse����ϴ���,losslayer������ǰ����� 
					if (!block->m_activation || block->m_layerType == Layer::LayerType::LossLayer)continue;
					block->forwardMove();
				}
			}
		};
		/*ѵ���׶�
		feedIn:��������
		lableAns:<AddLossLayer�����ص�id����,��ǵ���ȷ��>
		ѵ���׶�ͬʱ���Զ�ִ�з��򴫲�����
		*/
		void train(const Algebra::CMatrix& feedIn, std::map<int,Algebra::CMatrix>& lableAns) {
			predict(feedIn);//ִ���������
			for (auto lossLayer : m_lossLayer) {
				const int& lossId = lossLayer.first;
				auto lossPtr = lossLayer.second;
				if (lossPtr->m_layerType != Layer::LayerType::LossLayer)throw std::runtime_error("err: loss map contains other layerTypes...");
				if (lableAns.find(lossId) != lableAns.end()) {
					//ӵ�ж�Ӧ��ѵ������ ,����м���
					lossPtr->forwardMove(lableAns[lossId]);
				//	loosPtr
				}
			}
			//���򴫲�
			_backward();
		}

		void init() {
			for (auto layer : m_layer) {
				int blockCount = 0;
				for (auto block : layer.second) {
					//����
					block->m_alias = std::make_pair(layer.first, blockCount++);//layer + block
					//����activate��ʶ(����)
					auto frontConnectLayer = block->m_frontConnectLayer;
					for (auto front : frontConnectLayer) {
						if (!front->m_activation) {
							block->setActivate(false);
						}
					}
					
				}
			}
		}
		////����ִ�г�ʼ������
		//void setInit(const bool& init = true) {
		//	m_bInit = init;
		//}
	private:
		void _addLayer(std::list<std::shared_ptr<Layer::SunshineBaseLayer>>frontBlockIn, std::shared_ptr<Layer::SunshineBaseLayer> addBlock) {
			int deepest = -9999;
			for (auto i : frontBlockIn) {
				i->m_BackConnectLayer.push_back(addBlock);
				addBlock->m_frontConnectLayer.push_back(i);
				if (i->m_alias.first > deepest) {
					deepest = i->m_alias.first;
				}
			}
			//��ǰ����ѡ���������
			addBlock->m_alias = std::make_pair(deepest,-9999);
			m_layer[deepest + 1].push_back(addBlock);
			if (addBlock->m_lossLayerType != SunshineFrame::Layer::LossLayerType::Undefined) {
				m_lossLayer[addBlock->m_iLossID] = addBlock;
			}
		}
		void _addLayer(std::shared_ptr<Layer::SunshineBaseLayer>addBlock) {
			addBlock->m_alias = std::make_pair(0, -9999);
			m_layer[0].push_back(addBlock);
			if (addBlock->m_lossLayerType != SunshineFrame::Layer::LossLayerType::Undefined) {
				m_lossLayer[addBlock->m_iLossID] = addBlock;
			}
		}
		void _backward() {
			for (auto itr_layer = m_layer.rbegin(); itr_layer != m_layer.rend(); ++itr_layer) {
				for (auto block : itr_layer->second) {
					//���򴫲���losslayer��ʼ���д���
					block->backwardMove();
				}
			}
		};

		/*
		���������Ϣ:���ڽ�ѵ���õ�����������б���
		*/
		void _saveEncode(std::ofstream& ofs) {
			//��ܸ������岢��ÿһ�������Ȩֵ���������н���
			//ÿһ��block��$���зָ� ��|�ָ�alias,��βʹ��@���зָ� eg:wwwwwwww&0,1|xxxxxxx@wwwwwww&0,2|xxxxx@    
			//����0��1block�Ĳ���xxxxxxx�����ɸò����н���,wwww�Ĳ���Ϊ���฽��ֵ�����ܲ��ֽ��н���,���ڶ�������߶�ÿһ��block���и�����ϸ��˵��-�������version��Ϣ�ȣ�
			//Ŀǰ��δ�õ�@���ݣ������Ƕ��´�Э��
			for (auto layer : m_layer) {
				for (auto block : layer.second) {
					ofs << "&" << block->getAlias().first << "," << block->getAlias().second << "|";
					block->savePara(ofs);
					ofs << "@";
					//add some more para
				}
			}
		}
		/*��start��ʼ�� ��end���������ʹ��slice
		eg��"wwwwwwww&0,1|xxxxxxx@wwwwwww&0,2|xxxxx@dsadsadad&0,3|fffffff@dsadad";
		�����layer:0 block: 1 xxxxxxx
			  layer:0 block: 2 xxxxx
			  layer:0 block: 3 fffffff
		*/

		void _loadEncode(const std::string& is, const char start = '&', const char end = '@', const char slice = '|') {
			//�������鿴_saveEncode
			std::string buf(is.capacity(), '\n');
			std::istringstream ss(is);
			while (ss.getline(&buf[0], is.capacity(), '&')) {
				ss.getline(&buf[0], is.capacity(), '|');
				std::istringstream tmp(buf);
				std::string layerStr(400, '\n');
				std::string blockStr(400, '\n');
				tmp.getline(&layerStr[0], layerStr.capacity(), ',');
				tmp.getline(&blockStr[0], blockStr.capacity(), '\0');
				if (layerStr[0] == '\0' || blockStr[0] == '\0')continue;
				int layer = std::stoi(layerStr);
				int block = std::stoi(blockStr);
				if (layer < 0 || block < 0) {
					throw std::runtime_error("_loadEncode Err : layer and block number can't less 0 -- layer : " + std::to_string(layer) + " block: " + std::to_string(block));
				}
				ss.getline(&buf[0], is.capacity(), '@');
				//std::cout << buf.c_str() << "\n";
				//����frame��layer��block�Ŷ�Ӧ������
				if (m_layer.find(layer) != m_layer.end()) {
					auto specified_layer = m_layer[layer];
					//int idxBlock = 0;
					for (auto i : specified_layer)
					{
						if (i->m_alias.second== block) {
							i->loadPara(buf);
							break;
						}
					}
				}
				else {
					throw std::runtime_error("Err: _loadEncode failer.. no such layer number" + std::to_string(layer));
				}
			}
		}



	private:
		/*���в���Ϣ
		<��� ����ָ��>>
		*/
		std::map<int, std::list<std::shared_ptr<Layer::SunshineBaseLayer>>> m_layer;
		/*
		<lossLayer ID, loseLayָ��>
		ע��LossLayerID��AddLossLayer�������з��� 
		*/
		std::map<int, std::shared_ptr<Layer::SunshineBaseLayer>>m_lossLayer;
		int m_lossIdValid; //����loss ID�ķ���
		/*
		��ʼ����ʶ����һ�����л���ж�Ӧ��ʼ������
		*/
		bool m_bInit;

	};
}