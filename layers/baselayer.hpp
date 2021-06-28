#pragma once
#include <sstream>
#include <list>
#include "CMatrix.hpp"

namespace SunshineFrame {
	class Sunshine;
	namespace Layer {
		//ÿ����һ��layerType�ǵ�ͬ��
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
			MSE,//���������ʧ 
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
		ÿһ�����������������ϵ����Ҫ���棬��ô��������loadPara��saveparaͬʱ��������Ҫ���û����loadpara��savepara
		*/
		class SunshineBaseLayer {
			friend class Sunshine;//��namespace�޷�������namespace�����ݣ���Ҫ�ڸ�namespace��һ������
			//��������Ļ��ֱ��
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
			���������ڱ����Լ���ȡѵ������ʹ��
			para:��Ҫ���д���Ĳ�������
			���ز�������ȵ��û�������(�ú������뱻����)
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
			ofs:����д�Ķ���
			���ز�������ȵ��û�������(�ú������뱻����)
			*/
			virtual void savePara(std::ostream& ofs) {
				ofs << STARTCHAR << layerType2Str(m_layerType) << SPLITCHAR << lossLayerType2Str(m_lossLayerType) << SPLITCHAR << m_iLossID << SPLITCHAR << m_layerLearningRate << SPLITCHAR <<
					m_activation << SPLITCHAR << m_freezen << SPLITCHAR << m_usrAlias << ENDCHAR;
			}
			/*
			  ���ܣ������������¾���
			  feedIn:��������ľ���
			*/
			virtual void calBackUpdateMat(const Algebra::CMatrix& loss) {
				throw std::runtime_error("Must implete calBackUpdateMat func" + alias2String());
			}
			/*
			����:���º�������¾��������һ��calBackUpdateMat�����������ֵΪ׼��
			���ز�������ȵ��û�������(���������Ҫ�����¸ú������뱻���أ�
			*/
			virtual void updateBackMat(const Algebra::CMatrix& loss) {
				if (m_freezen || !m_activation)return;
				//throw std::runtime_error("Must implete updateBackMat func" + alias2String());
			};
			/*
			����: ǰ�򴫲�
			feedIn:����ò������(��Ϊ�ײ�ʹ�� )
			���ز�������ȵ��û�������(�ú������뱻����)
			*/
			virtual void forwardMove(const Algebra::CMatrix& feedIn) {
				if (!m_activation)return;
				//throw std::runtime_error("Must implete forwardMove(feedIn) func " + alias2String());
			}

			/*
			����: ǰ�򴫲�
			���ز�������ȵ��û�������
			*/
			virtual void forwardMove() {
				if (!m_activation)return;
				//throw std::runtime_error("Must implete forwardMove func" + alias2String());

			}

			/*
			����: ���򴫲�
			*/
			virtual void backwardMove() {
				if (!m_activation)return;
			}
			/*
			����: ���ü������
			*/
			void setActivate(const bool& ac) {
				m_activation = ac;
			}
			/*
			����: ���ö������
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
			//����m_front2backMat�� m_back2frontMat��Ҫͨ��ָ����з��ʣ������޷�����protected��
			Algebra::CMatrix m_front2backMat;//ǰ�򴫲�����(���ڴ������layer�����ݣ�ʹ��forwardMove����ó� ) eg: y=w1*x1 + w0 * x0 �е���y��ɵľ���, ����ʹ�������з�ʽ,��������жѵ�
			Algebra::CMatrix m_back2frontMat;//���������(���ں��������Ҫ����ǰ��layer�ľ���ʹ��backMove����ó�)
		protected:
			std::string findValue(std::string& para, char key) {
			//�Ὣkey��ǰ�����ݽ��н�ȡ��ͬʱ���޸�para������key֮��Ĳ��֣�������key֮ǰ������
			//eg: 	std::string a = "dsadawjklqw!#$[fc:mse:3.1415926:1:0]fdsfs&qwewqs|3dasdsdas@";
			// key = "["  => a == fc:mse:3.1415926:1:0]fdsfs&qwewqs|3dasdsdas@ ����"dsadawjklqw!#$
			//
				size_t idx = para.find_first_of(key);
				if (idx != std::string::npos) {
					std::string out = para.substr(0, idx);
					para = para.substr(idx + 1, std::string::npos);
					return std::move(out);
				}
				throw std::runtime_error("find value shutdown:" + para + " do not contains: " + key);
			};
			LayerType m_layerType;//������ 
			LossLayerType m_lossLayerType;//�������ʧ������,���������ΪLossLayer�Ļ�
			/*
			 ��ţ����:���п����ִ��init֮��Ż����
			*/
			std::pair<int, int> m_alias;
			std::string m_usrAlias;
			/*
			 ��ʧ����ID��
			*/
			int m_iLossID;
			std::list<std::shared_ptr<SunshineBaseLayer>>m_frontConnectLayer;//ǰ�����Ӳ�,��������ǰ��Ĳ�
			std::list<std::shared_ptr<SunshineBaseLayer>>m_BackConnectLayer;//�������Ӳ㣬�������Ӻ����Ĳ�
			Algebra::MatrixDataType m_layerLearningRate;//��ѧϰ�ʣ�����ָ�������ѧϰ��
		private:
			/*
				m_activation��m_freezen����:m_activation��ضϴ������̣���freezen����ضϴ������̣��������ý������ǵĴ��ݹ���֮��Ȩֵ���ٽ��и���
			*/
			bool m_activation;//�˲��Ƿ�ʹ���Լ����� true:�����ʹ�� ,activation�д���Ч����һ����Ϊ0��������ص�activate����ʧЧ����init����ִ�к���Ч

			bool m_freezen;//�˲��Ƿ񱻶��� true:���� 

		};

	}
}