#pragma once
#include <sstream>
#include <list>
#include "CMatrix.hpp"

namespace SunshineFrame {
	class Sunshine;
	namespace Layer {
		//每增加一个layerType记得同步
		enum class LayerType {
			FullyConnect,
			Relu,
			LossLayer,
			Undefined
		};
		LayerType str2layerType(const std::string& in) {
			if (!in.compare("fc")) {
				return LayerType::FullyConnect;
			}
			else if (!in.compare("lossLayer")) {
				return LayerType::LossLayer;
			}
			else if(!in.compare("relu"))
			{
				return LayerType::Relu;
			}
			else if (!in.compare("undefined")) 
			{
				return LayerType::Undefined;
			}
			else {
			}
			throw std::runtime_error("Err str2layertype not founded..." + in);
		}
		std::string layerType2Str(const LayerType& lt) {
			std::string retName = "";
			switch (lt)
			{
			case LayerType::FullyConnect:
				retName = "fc";
				break;
			case LayerType::LossLayer:
				retName = "lossLayer";
				break;
			case LayerType::Relu:
				retName = "relu";
				break;
			case LayerType::Undefined:
				retName = "undefined";
				break;
			default:
				throw std::runtime_error(" layerType2Str Err: default type...");
				break;
			}
			return retName;
		};

		enum class LossLayerType {
			MSE,//均方误差损失 
			Undefined,//
		};
		std::string lossLayerType2Str(const LossLayerType& llt) {
			std::string retName = "";
			switch (llt)
			{
			case LossLayerType::MSE:
				retName = "mse";
				break;
			case LossLayerType::Undefined:
				retName = "undefined";
				break;
			default:
				throw std::runtime_error(" layerType2Str Err: default type...");
				break;
			}
			return retName;
		};
		LossLayerType str2lossLayerType(const std::string& str) {
			if (!str.compare("mse")) 
			{
				return LossLayerType::MSE;
			}
			else if (!str.compare("undefined")) {
				return LossLayerType::Undefined;
			}
			throw std::runtime_error("Err str2lossLayertype not founded..." + str);
		}

		///////////////////////////////////////////////////////////////////////////
		/*
		每一个重载倘若有自身的系数需要保存，那么必须重载loadPara和savepara同时，首先需要调用基类的loadpara和savepara
		*/
		class SunshineBaseLayer {
			friend class Sunshine;//子namespace无法看到父namespace的内容，需要在父namespace做一个申明
			//保存参数的划分标记
			const char SPLITCHAR = ':';
			const char STARTCHAR = '[';
			const char ENDCHAR = ']';
		public:
			SunshineBaseLayer(const LayerType& type, const double& learnRate = 0.01, std::string user_alias = "undefined", bool activate = true, bool freeze = false) {
				m_layerType = type;
				m_lossLayerType = LossLayerType::Undefined;
				m_frontConnectLayer.clear();
				m_BackConnectLayer.clear();
				m_activation = activate;
				m_freezen = freeze;
				m_layerLearningRate = learnRate;
				m_alias = { -1,-1 };
				m_usrAlias = user_alias;
				m_iLossID = -1;
			};
			~SunshineBaseLayer() {
			};
		public:
			/*
			此两项用于保存以及读取训练参数使用
			para:需要进行处理的参数内容
			重载层必须首先调用基类内容(该函数必须被重载)
			*/
			virtual void loadPara(std::string& para) {
				std::string&& tmp = findValue(para, STARTCHAR);
				tmp = findValue(para, SPLITCHAR);
				m_layerType = str2layerType(tmp);
				tmp = findValue(para, SPLITCHAR);
				m_lossLayerType = str2lossLayerType(tmp);
				tmp = findValue(para, SPLITCHAR);
				m_iLossID = std::atoi(tmp.c_str());
				tmp = findValue(para, SPLITCHAR);
				m_layerLearningRate = std::strtod(tmp.c_str(),nullptr);
				tmp = findValue(para, SPLITCHAR);
				m_activation = std::atoi(tmp.c_str());
				tmp = findValue(para, SPLITCHAR);
				m_freezen = std::atoi(tmp.c_str());
				tmp = findValue(para, ENDCHAR);
				m_usrAlias = tmp;

			}
			virtual void showWeight() {
				std::cout << "alisa:  " << alias2String() << " user_alias: " << m_usrAlias << "\n";
				return;
			}
			/*
			ofs:用于写的对象
			重载层必须首先调用基类内容(该函数必须被重载)
			*/
			virtual void savePara(std::ostream& ofs) {
				ofs << STARTCHAR << layerType2Str(m_layerType) << SPLITCHAR << lossLayerType2Str(m_lossLayerType) << SPLITCHAR << m_iLossID << SPLITCHAR << m_layerLearningRate << SPLITCHAR <<
					m_activation << SPLITCHAR << m_freezen << SPLITCHAR << m_usrAlias << ENDCHAR;
			}
			/*
			  功能：计算后项传播更新矩阵
			  feedIn:后项输入的矩阵
			*/
			virtual void calBackUpdateMat(const Algebra::CMatrix& loss) {
				throw std::runtime_error("Must implete calBackUpdateMat func" + alias2String());
			}
			/*
			功能:更新后项传播更新矩阵（以最近一次calBackUpdateMat所计算出来的值为准）
			重载层必须首先调用基类内容(如果参数需要被更新该函数必须被重载）
			*/
			virtual void updateBackMat(const Algebra::CMatrix& loss) {
				if (m_freezen || !m_activation)return;
				//throw std::runtime_error("Must implete updateBackMat func" + alias2String());
			};
			/*
			功能: 前向传播
			feedIn:传入该层的数据(作为首层使用 )
			重载层必须首先调用基类内容(该函数必须被重载)
			*/
			virtual void forwardMove(const Algebra::CMatrix& feedIn) {
				if (!m_activation)return;
				//throw std::runtime_error("Must implete forwardMove(feedIn) func " + alias2String());
			}

			/*
			功能: 前向传播
			重载层必须首先调用基类内容
			*/
			virtual void forwardMove() {
				if (!m_activation)return;
				//throw std::runtime_error("Must implete forwardMove func" + alias2String());

			}

			/*
			功能: 后向传播
			*/
			virtual void backwardMove() {
				if (!m_activation)return;
			}
			/*
			功能: 设置激活参数
			*/
			void setActivate(const bool& ac) {
				m_activation = ac;
			}
			/*
			功能: 设置冻结参数
			*/
			void setFreezen(const bool& fz) {
				m_freezen = fz;
			}
			inline Algebra::CMatrix getFront2BackMat() {
				return m_front2backMat;
			}
			inline Algebra::CMatrix getBack2FrontMat() {
				return m_back2frontMat;
			}
			inline std::string alias2String() {
				std::ostringstream ss;
				ss << "Layer: " << m_alias.first << " block: " << m_alias.second;
				return ss.str();
			}
			inline std::string lossID2String() {
				std::ostringstream ss;
				ss << "lossId: " << m_iLossID << " layer: " << m_alias.first << " block: " << m_alias.second;
				return ss.str();
			}
			inline void setLayerType(const LayerType& a) {
				m_layerType = a;
			}
			inline LayerType getLayerType() {
				return m_layerType;
			}
			inline void setLossLayerType(const LossLayerType& a) {
				m_lossLayerType = a;
			}
			inline LossLayerType getLossLayerType() {
				return m_lossLayerType;
			}
			inline void setAlias(const std::pair<int, int>& a) {
				m_alias = a;
			}
			inline std::pair<int, int> getAlias() {
				return m_alias;
			}
			inline int getLossID() {
				return m_iLossID;
			}
		public:
			//由于m_front2backMat， m_back2frontMat需要通过指针进行访问，所以无法放入protected中
			Algebra::CMatrix m_front2backMat;//前向传播矩阵(用于传入后续layer的数据，使用forwardMove计算得出 ) eg: y=w1*x1 + w0 * x0 中的由y组成的矩阵, 数据使用行排列方式,多个数据列堆叠
			Algebra::CMatrix m_back2frontMat;//后项传播矩阵(用于后项传播中需要传入前向layer的矩阵，使用backMove计算得出)
		protected:
			std::string findValue(std::string& para, char key) {
			//会将key以前的内容进行截取，同时会修改para（保留key之后的部分），返回key之前的内容
			//eg: 	std::string a = "dsadawjklqw!#$[fc:mse:3.1415926:1:0]fdsfs&qwewqs|3dasdsdas@";
			// key = "["  => a == fc:mse:3.1415926:1:0]fdsfs&qwewqs|3dasdsdas@ 返回"dsadawjklqw!#$
			//
				size_t idx = para.find_first_of(key);
				if (idx != std::string::npos) {
					std::string out = para.substr(0, idx);
					para = para.substr(idx + 1, std::string::npos);
					return std::move(out);
				}
				throw std::runtime_error("find value shutdown:" + para + " do not contains: " + key);
			};
			LayerType m_layerType;//层类型 
			LossLayerType m_lossLayerType;//具体的损失层类型,如果层类型为LossLayer的话
			/*
			 层号，块号:其中块号在执行init之后才会更新
			*/
			std::pair<int, int> m_alias;
			std::string m_usrAlias;
			/*
			 损失函数ID号
			*/
			int m_iLossID;
			std::list<std::shared_ptr<SunshineBaseLayer>>m_frontConnectLayer;//前向连接层,用于连接前面的层
			std::list<std::shared_ptr<SunshineBaseLayer>>m_BackConnectLayer;//后向连接层，用于连接后续的层
			Algebra::MatrixDataType m_layerLearningRate;//层学习率，用于指导整层的学习率
		private:
			/*
				m_activation，m_freezen区别:m_activation会截断传播过程，而freezen不会截断传播过程，他的作用仅仅是是的传递过程之中权值不再进行更新
			*/
			bool m_activation;//此层是否被使用以及激活 true:激活并被使用 ,activation有传播效果，一旦其为0，与它相关的activate都会失效，在init函数执行后生效

			bool m_freezen;//此层是否被冻结 true:冻结 

		};

	}
}