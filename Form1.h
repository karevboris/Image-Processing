#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <fstream>
#include <ctime>
#include <vector>
#include <iterator>
#include <omp.h>
#include <vcclr.h>
#include <sys/types.h>

#define WIDTH 1024
#define HEIGHT 768

#define UNIFORM 1
#define K_MEANS 2

using namespace cv;
using namespace std;

struct statistics {
	double meanX;
	double meanY;
	double deviationX;
	double deviationY;

	statistics() {
		meanX = 0;
		meanY = 0;
		deviationX = 0;
		deviationY = 0;
	}
};

struct matrix {
	Mat *mat;
	uchar *data;
	int n;
	matrix() {}
};

int SIZE = WIDTH * HEIGHT;
int DEEP = 1;
int LEVEL_NUM = 8;
int const NUM_FEATURES = 8;
double EPS = 0.045;
double EPS2 = 0.045;
matrix mat;
double *vec = new double[NUM_FEATURES];
uchar* memblock = new uchar[WIDTH*HEIGHT*1];
uchar* testMemblock = new uchar[WIDTH*HEIGHT*1];
uchar* trainMemblock = new uchar[WIDTH*HEIGHT*1];
double *errors = new double[WIDTH*HEIGHT*1];
uchar* testingMarks = new uchar[HEIGHT * WIDTH];
int value = EPS * 1000;
bool isFinished = false;
vector<vector<cv::Point>> vertices;
int index = 0;
bool first = true;
uchar* clusters;
bool quantflag = false;
int COUNT = 215;

namespace CppCLRWinformsProjekt {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;

	public ref class Form1 : public System::Windows::Forms::Form {
	public: delegate void ChangeThresholdDelegate(int Th, void* param);
	public: ChangeThresholdDelegate^ ChangeThresholdDelegateInstance;
	public: cv::TrackbarCallback ChangeThresholdCallbackPointer;

	public: delegate void ChangeThresholdDelegate2(int Th, void* param);
	public: ChangeThresholdDelegate2^ ChangeThresholdDelegateInstance2;
	public: cv::TrackbarCallback ChangeThresholdCallbackPointer2;

	public: delegate void onMouseDelegate(int event, int x, int y, int flags, void* param);
	public: onMouseDelegate^ onMouseDelegateInstance;
	public: cv::MouseCallback onMouseCallbackPointer;

	private: System::Windows::Forms::Label^ label1;
	private: System::Windows::Forms::Label^ label2;
	private: System::Windows::Forms::TextBox^ textBox1;
	private: System::Windows::Forms::CheckBox^ checkBox1;
	private: System::Windows::Forms::CheckBox^ checkBox2;
	private: System::Windows::Forms::NumericUpDown^ numericUpDown2;
	private: System::Windows::Forms::Label^ label3;
	private: System::Windows::Forms::CheckBox^ checkBox3;
	private: System::Windows::Forms::Label^ label4;
	private: System::Windows::Forms::NumericUpDown^ numericUpDown3;

	private: System::Windows::Forms::NumericUpDown^ numericUpDown5;
	private: System::Windows::Forms::NumericUpDown^ numericUpDown6;
	private: System::Windows::Forms::Label^ label5;

	private: System::Windows::Forms::Button^ button3;
	private: System::Windows::Forms::DataVisualization::Charting::Chart^ chart1;
	private: System::Windows::Forms::ComboBox^ comboBox2;
	private: System::Windows::Forms::ComboBox^ comboBox3;

	private: System::Windows::Forms::Label^ label7;
	private: System::Windows::Forms::Label^ label6;

	private: System::Windows::Forms::CheckBox^ checkBox4;

	public:
		Form1(void)
		{
			InitializeComponent();

			ChangeThresholdDelegateInstance = gcnew ChangeThresholdDelegate(this, &Form1::ChangeThreshold);
			IntPtr delegatePointer = System::Runtime::InteropServices::Marshal::GetFunctionPointerForDelegate(ChangeThresholdDelegateInstance);
			ChangeThresholdCallbackPointer = reinterpret_cast<cv::TrackbarCallback>(delegatePointer.ToPointer());

			ChangeThresholdDelegateInstance2 = gcnew ChangeThresholdDelegate2(this, &Form1::ChangeThreshold2);
			IntPtr delegatePointer2 = System::Runtime::InteropServices::Marshal::GetFunctionPointerForDelegate(ChangeThresholdDelegateInstance2);
			ChangeThresholdCallbackPointer2 = reinterpret_cast<cv::TrackbarCallback>(delegatePointer2.ToPointer());

			onMouseDelegateInstance = gcnew onMouseDelegate(this, &Form1::onMouse);
			IntPtr onMouseDelegatePointer = System::Runtime::InteropServices::Marshal::GetFunctionPointerForDelegate(onMouseDelegateInstance);
			onMouseCallbackPointer = reinterpret_cast<cv::MouseCallback>(onMouseDelegatePointer.ToPointer());
		}

	protected: ~Form1()
		{
			if (components)
			{
				delete components;
			}
		}
	private: System::Windows::Forms::Button^ button1;
	private: System::Windows::Forms::ComboBox^ comboBox1;

	public: System::Void ChangeThreshold(int Th, void* param) {
		if (checkBox3->Checked) {
			EPS = double(Th) / 100000;
		}
		else {
			EPS = double(Th) / 100000;
		}
		draw();
	}

	public: System::Void ChangeThreshold2(int Th, void* param) {
		double procent = (double)numericUpDown5->Value;
		if (checkBox3->Checked) {
			EPS2 = double(Th) / (100000 / procent);
		}
		else {
			EPS2 = double(Th) / (100000 / procent);
		}
		draw2();
	}

	public: System::Void onMouse(int event, int x, int y, int flags, void* param) {
		if (event == EVENT_LBUTTONDOWN) {
			textBox1->AppendText("x = " + x + "; y = " + y + "\r\n");

			LEVEL_NUM = (int)numericUpDown3->Value;
			int BLOCK_SIZE = int(numericUpDown2->Value);
			uchar* tmp = new uchar[BLOCK_SIZE * BLOCK_SIZE];
			float* standart = new float[LEVEL_NUM * LEVEL_NUM];

			quantizationGreyLevel(memblock, WIDTH, HEIGHT, DEEP, LEVEL_NUM, comboBox3->Text->ToString());

			for (int k = 0; k < BLOCK_SIZE; k++)
				for (int m = 0; m < BLOCK_SIZE; m++) {
					tmp[k * BLOCK_SIZE + m] = memblock[(y - BLOCK_SIZE / 2 + k) * WIDTH + x - BLOCK_SIZE / 2 + m];
				}
			glcmHor(tmp, BLOCK_SIZE, BLOCK_SIZE, 0, LEVEL_NUM, standart);
			glcmNormalize(standart, LEVEL_NUM);

			Mat record = Mat(HEIGHT, WIDTH, CV_64F);
			calculateByScal3(standart, testMemblock, record);
			writeCSV("Data/single.csv", record);
		}

		if (event == EVENT_MBUTTONDOWN) {
			chart1->Series[0]->Points->Clear();
			chart1->ChartAreas[0]->AxisX->Minimum = 0;
			if (LEVEL_NUM != numericUpDown3->Value) {
				quantflag = false;
			}
			LEVEL_NUM = (int)numericUpDown3->Value;
			int BLOCK_SIZE = int(numericUpDown2->Value);
			int* numbers = new int[LEVEL_NUM];

			if (!quantflag && LEVEL_NUM != 256) {
				quantizationGreyLevel(memblock, WIDTH, HEIGHT, DEEP, LEVEL_NUM, comboBox3->Text->ToString());
				quantflag = true;
			}

			for (int i = 0; i < LEVEL_NUM; i++) numbers[i] = 0;

			int index;
			if (BLOCK_SIZE != 0) {
				for (int k = 0; k < BLOCK_SIZE; k++)
					for (int m = 0; m < BLOCK_SIZE; m++) {
						index =  memblock[(y - BLOCK_SIZE / 2 + k) * WIDTH + x - BLOCK_SIZE / 2 + m];
						numbers[index]++;
					}
			}
			else {
				for (int i = 0; i < HEIGHT; i++)
					for (int j = 0; j < WIDTH; j++) {
						index = memblock[i * WIDTH + j];
						numbers[index]++;
					}
			}

			for (int i = 0; i < LEVEL_NUM; i++) {
				chart1->Series[0]->Points->AddXY(i, numbers[i]);
			}

			double* params = new double[6];
			getHistParams(numbers, LEVEL_NUM, params);
			delete[] params;
		}

		if (event == EVENT_RBUTTONDOWN) {
			int level_num = (int)numericUpDown3->Value;
			int BLOCK_SIZE = int(numericUpDown2->Value);
			uchar* tmp = new uchar[BLOCK_SIZE * BLOCK_SIZE];
			double* numbers = new double[level_num];

			if (level_num != 256) quantizationGreyLevel(trainMemblock, WIDTH, HEIGHT, 1, level_num, comboBox3->Text->ToString());

			for (int i = 0; i < level_num; i++) numbers[i] = 0;

			for (int k = 0; k < BLOCK_SIZE; k++)
				for (int m = 0; m < BLOCK_SIZE; m++) {
					tmp[k * BLOCK_SIZE + m] = trainMemblock[(y - BLOCK_SIZE / 2 + k) * WIDTH + x - BLOCK_SIZE / 2 + m];
					numbers[tmp[k * BLOCK_SIZE + m]]++;
				}

			for (int i = 0; i < level_num; i++) {
				numbers[i] /= BLOCK_SIZE * BLOCK_SIZE;
			}

			calculateByHist(memblock, WIDTH, HEIGHT, 1, numbers);

			delete[] tmp, numbers;
		}
	}

	private:
		System::ComponentModel::Container ^components;

#pragma region Windows Form Designer generated code
		void InitializeComponent(void)
		{
			System::Windows::Forms::DataVisualization::Charting::ChartArea^ chartArea1 = (gcnew System::Windows::Forms::DataVisualization::Charting::ChartArea());
			System::Windows::Forms::DataVisualization::Charting::Legend^ legend1 = (gcnew System::Windows::Forms::DataVisualization::Charting::Legend());
			System::Windows::Forms::DataVisualization::Charting::Series^ series1 = (gcnew System::Windows::Forms::DataVisualization::Charting::Series());
			this->button1 = (gcnew System::Windows::Forms::Button());
			this->comboBox1 = (gcnew System::Windows::Forms::ComboBox());
			this->label1 = (gcnew System::Windows::Forms::Label());
			this->label2 = (gcnew System::Windows::Forms::Label());
			this->textBox1 = (gcnew System::Windows::Forms::TextBox());
			this->checkBox1 = (gcnew System::Windows::Forms::CheckBox());
			this->checkBox2 = (gcnew System::Windows::Forms::CheckBox());
			this->numericUpDown2 = (gcnew System::Windows::Forms::NumericUpDown());
			this->label3 = (gcnew System::Windows::Forms::Label());
			this->checkBox3 = (gcnew System::Windows::Forms::CheckBox());
			this->label4 = (gcnew System::Windows::Forms::Label());
			this->numericUpDown3 = (gcnew System::Windows::Forms::NumericUpDown());
			this->numericUpDown5 = (gcnew System::Windows::Forms::NumericUpDown());
			this->numericUpDown6 = (gcnew System::Windows::Forms::NumericUpDown());
			this->label5 = (gcnew System::Windows::Forms::Label());
			this->button3 = (gcnew System::Windows::Forms::Button());
			this->chart1 = (gcnew System::Windows::Forms::DataVisualization::Charting::Chart());
			this->checkBox4 = (gcnew System::Windows::Forms::CheckBox());
			this->comboBox2 = (gcnew System::Windows::Forms::ComboBox());
			this->comboBox3 = (gcnew System::Windows::Forms::ComboBox());
			this->label7 = (gcnew System::Windows::Forms::Label());
			this->label6 = (gcnew System::Windows::Forms::Label());
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown2))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown3))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown5))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown6))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->chart1))->BeginInit();
			this->SuspendLayout();
			// 
			// button1
			// 
			this->button1->Location = System::Drawing::Point(200, 79);
			this->button1->Name = L"button1";
			this->button1->Size = System::Drawing::Size(75, 23);
			this->button1->TabIndex = 0;
			this->button1->Text = L"Run";
			this->button1->UseVisualStyleBackColor = true;
			this->button1->Click += gcnew System::EventHandler(this, &Form1::button1_Click);
			// 
			// comboBox1
			// 
			this->comboBox1->FormattingEnabled = true;
			this->comboBox1->Items->AddRange(gcnew cli::array< System::Object^  >(3) { L"QUAD", L"MAX", L"ABS" });
			this->comboBox1->Location = System::Drawing::Point(70, 76);
			this->comboBox1->Name = L"comboBox1";
			this->comboBox1->Size = System::Drawing::Size(120, 21);
			this->comboBox1->TabIndex = 1;
			this->comboBox1->Text = L"QUAD";
			// 
			// label1
			// 
			this->label1->AutoSize = true;
			this->label1->Location = System::Drawing::Point(5, 79);
			this->label1->Name = L"label1";
			this->label1->Size = System::Drawing::Size(52, 13);
			this->label1->TabIndex = 2;
			this->label1->Text = L"Error type";
			// 
			// label2
			// 
			this->label2->AutoSize = true;
			this->label2->Location = System::Drawing::Point(5, 9);
			this->label2->Name = L"label2";
			this->label2->Size = System::Drawing::Size(25, 13);
			this->label2->TabIndex = 3;
			this->label2->Text = L"Eps";
			// 
			// textBox1
			// 
			this->textBox1->Location = System::Drawing::Point(8, 199);
			this->textBox1->Multiline = true;
			this->textBox1->Name = L"textBox1";
			this->textBox1->ScrollBars = System::Windows::Forms::ScrollBars::Vertical;
			this->textBox1->Size = System::Drawing::Size(260, 172);
			this->textBox1->TabIndex = 5;
			// 
			// checkBox1
			// 
			this->checkBox1->AutoSize = true;
			this->checkBox1->Location = System::Drawing::Point(70, 176);
			this->checkBox1->Name = L"checkBox1";
			this->checkBox1->Size = System::Drawing::Size(41, 17);
			this->checkBox1->TabIndex = 6;
			this->checkBox1->Text = L"Bin";
			this->checkBox1->UseVisualStyleBackColor = true;
			// 
			// checkBox2
			// 
			this->checkBox2->AutoSize = true;
			this->checkBox2->Location = System::Drawing::Point(130, 176);
			this->checkBox2->Name = L"checkBox2";
			this->checkBox2->Size = System::Drawing::Size(66, 17);
			this->checkBox2->TabIndex = 7;
			this->checkBox2->Text = L"Equalize";
			this->checkBox2->UseVisualStyleBackColor = true;
			// 
			// numericUpDown2
			// 
			this->numericUpDown2->Increment = System::Decimal(gcnew cli::array< System::Int32 >(4) { 2, 0, 0, 0 });
			this->numericUpDown2->Location = System::Drawing::Point(70, 30);
			this->numericUpDown2->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 256, 0, 0, 0 });
			this->numericUpDown2->Name = L"numericUpDown2";
			this->numericUpDown2->Size = System::Drawing::Size(120, 20);
			this->numericUpDown2->TabIndex = 8;
			this->numericUpDown2->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 29, 0, 0, 0 });
			// 
			// label3
			// 
			this->label3->AutoSize = true;
			this->label3->Location = System::Drawing::Point(5, 32);
			this->label3->Name = L"label3";
			this->label3->Size = System::Drawing::Size(55, 13);
			this->label3->TabIndex = 9;
			this->label3->Text = L"Block size";
			// 
			// checkBox3
			// 
			this->checkBox3->AutoSize = true;
			this->checkBox3->Location = System::Drawing::Point(215, 176);
			this->checkBox3->Name = L"checkBox3";
			this->checkBox3->Size = System::Drawing::Size(53, 17);
			this->checkBox3->TabIndex = 10;
			this->checkBox3->Text = L"Scala";
			this->checkBox3->UseVisualStyleBackColor = true;
			// 
			// label4
			// 
			this->label4->AutoSize = true;
			this->label4->Location = System::Drawing::Point(5, 55);
			this->label4->Name = L"label4";
			this->label4->Size = System::Drawing::Size(38, 13);
			this->label4->TabIndex = 12;
			this->label4->Text = L"Levels";
			// 
			// numericUpDown3
			// 
			this->numericUpDown3->Location = System::Drawing::Point(70, 53);
			this->numericUpDown3->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 256, 0, 0, 0 });
			this->numericUpDown3->Name = L"numericUpDown3";
			this->numericUpDown3->Size = System::Drawing::Size(120, 20);
			this->numericUpDown3->TabIndex = 26;
			this->numericUpDown3->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 16, 0, 0, 0 });
			// 
			// numericUpDown5
			// 
			this->numericUpDown5->DecimalPlaces = 5;
			this->numericUpDown5->ForeColor = System::Drawing::SystemColors::WindowText;
			this->numericUpDown5->Increment = System::Decimal(gcnew cli::array< System::Int32 >(4) { 2, 0, 0, 0 });
			this->numericUpDown5->Location = System::Drawing::Point(70, 7);
			this->numericUpDown5->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1000, 0, 0, 0 });
			this->numericUpDown5->Name = L"numericUpDown5";
			this->numericUpDown5->Size = System::Drawing::Size(120, 20);
			this->numericUpDown5->TabIndex = 14;
			this->numericUpDown5->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 131072 });
			// 
			// numericUpDown6
			// 
			this->numericUpDown6->Location = System::Drawing::Point(107, 147);
			this->numericUpDown6->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 200, 0, 0, 0 });
			this->numericUpDown6->Name = L"numericUpDown6";
			this->numericUpDown6->Size = System::Drawing::Size(83, 20);
			this->numericUpDown6->TabIndex = 15;
			this->numericUpDown6->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
			// 
			// label5
			// 
			this->label5->AutoSize = true;
			this->label5->Location = System::Drawing::Point(5, 149);
			this->label5->Name = L"label5";
			this->label5->Size = System::Drawing::Size(85, 13);
			this->label5->TabIndex = 16;
			this->label5->Text = L"Number of slides";
			// 
			// button3
			// 
			this->button3->Location = System::Drawing::Point(200, 7);
			this->button3->Name = L"button3";
			this->button3->Size = System::Drawing::Size(75, 23);
			this->button3->TabIndex = 18;
			this->button3->Text = L"Open file";
			this->button3->UseVisualStyleBackColor = true;
			this->button3->Click += gcnew System::EventHandler(this, &Form1::button3_Click);
			// 
			// chart1
			// 
			chartArea1->Name = L"ChartArea1";
			this->chart1->ChartAreas->Add(chartArea1);
			legend1->Enabled = false;
			legend1->Name = L"Legend1";
			this->chart1->Legends->Add(legend1);
			this->chart1->Location = System::Drawing::Point(281, 7);
			this->chart1->Name = L"chart1";
			series1->ChartArea = L"ChartArea1";
			series1->IsVisibleInLegend = false;
			series1->Legend = L"Legend1";
			series1->Name = L"Ãטסעמדנאללא";
			this->chart1->Series->Add(series1);
			this->chart1->Size = System::Drawing::Size(448, 364);
			this->chart1->TabIndex = 19;
			this->chart1->Text = L"chart1";
			// 
			// checkBox4
			// 
			this->checkBox4->AutoSize = true;
			this->checkBox4->Location = System::Drawing::Point(8, 176);
			this->checkBox4->Name = L"checkBox4";
			this->checkBox4->Size = System::Drawing::Size(44, 17);
			this->checkBox4->TabIndex = 20;
			this->checkBox4->Text = L"Hist";
			this->checkBox4->UseVisualStyleBackColor = true;
			// 
			// comboBox2
			// 
			this->comboBox2->FormattingEnabled = true;
			this->comboBox2->Items->AddRange(gcnew cli::array< System::Object^  >(5) {
				L"Test k-means", L"Display histogramm", L"Extract features",
					L"Expansion of regions", L"Testing tool"
			});
			this->comboBox2->Location = System::Drawing::Point(70, 122);
			this->comboBox2->Name = L"comboBox2";
			this->comboBox2->Size = System::Drawing::Size(120, 21);
			this->comboBox2->TabIndex = 21;
			this->comboBox2->Text = L"Extract features";
			// 
			// comboBox3
			// 
			this->comboBox3->FormattingEnabled = true;
			this->comboBox3->Items->AddRange(gcnew cli::array< System::Object^  >(2) { L"Uniform", L"K-Means" });
			this->comboBox3->Location = System::Drawing::Point(70, 99);
			this->comboBox3->Name = L"comboBox3";
			this->comboBox3->Size = System::Drawing::Size(120, 21);
			this->comboBox3->TabIndex = 22;
			this->comboBox3->Text = L"K-Means";
			// 
			// label7
			// 
			this->label7->AutoSize = true;
			this->label7->Location = System::Drawing::Point(5, 125);
			this->label7->Name = L"label7";
			this->label7->Size = System::Drawing::Size(31, 13);
			this->label7->TabIndex = 24;
			this->label7->Text = L"Task";
			// 
			// label6
			// 
			this->label6->AutoSize = true;
			this->label6->Location = System::Drawing::Point(5, 102);
			this->label6->Name = L"label6";
			this->label6->Size = System::Drawing::Size(59, 13);
			this->label6->TabIndex = 27;
			this->label6->Text = L"Quant type";
			// 
			// Form1
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(741, 383);
			this->Controls->Add(this->label6);
			this->Controls->Add(this->label7);
			this->Controls->Add(this->comboBox3);
			this->Controls->Add(this->comboBox2);
			this->Controls->Add(this->checkBox4);
			this->Controls->Add(this->chart1);
			this->Controls->Add(this->button3);
			this->Controls->Add(this->label5);
			this->Controls->Add(this->numericUpDown6);
			this->Controls->Add(this->numericUpDown5);
			this->Controls->Add(this->label4);
			this->Controls->Add(this->numericUpDown3);
			this->Controls->Add(this->checkBox3);
			this->Controls->Add(this->label3);
			this->Controls->Add(this->numericUpDown2);
			this->Controls->Add(this->checkBox2);
			this->Controls->Add(this->checkBox1);
			this->Controls->Add(this->textBox1);
			this->Controls->Add(this->label2);
			this->Controls->Add(this->label1);
			this->Controls->Add(this->comboBox1);
			this->Controls->Add(this->button1);
			this->Name = L"Form1";
			this->Text = L"Texture feature extractor";
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown2))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown3))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown5))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown6))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->chart1))->EndInit();
			this->ResumeLayout(false);
			this->PerformLayout();

		}

		int metric(int x, int y) {
			return abs(x - y);
		}

		void uniformQuantization(uchar* data, int width, int height, int deep, int clustNum) {
			float size = 256.0f / clustNum;
			for (int z = 0; z < deep; z++)
				for (int i = 0; i < height; i++)
					for (int j = 0; j < width; j++) {
						data[i * width + j + z * width * height] /= size;
					}
		}

		void commonQuantization(uchar* data, int width, int height, int deep, int clustNum) {
			float size = 256.0f / clustNum;
			for (int z = 0; z < deep; z++)
				for (int i = 0; i < height; i++)
					for (int j = 0; j < width; j++) {
						if (data[i * width + j + z * width * height] != 0) {
							data[i * width + j + z * width * height] /= size;
						}
						else {
							data[i * width + j + z * width * height] = 255;
						}
					}
		}

		void kMeansQuantization(uchar* data, int width, int height, int deep, int clustNum) {
			srand(time(0));
			int numIterationK = 10;
			uchar value = 0;
			long* sum = new long[clustNum];
			int* number = new int[clustNum];
			int minLimit = -1;
			int maxLimit = 256;

			for (int i = 0; i < clustNum; i++) {
				if (clustNum == 4) {
					clusters[0] = 100;
					clusters[1] = 110;
					clusters[2] = 120;
					clusters[3] = 130;
				}
				else {
					clusters[i] = i * (256 / clustNum);
				}
				number[i] = sum[i] = 0;
			}

			int iter = 0;

			while (iter < numIterationK) {
				for (int z = 0; z < deep; z++)
					for (int i = 0; i < height; i++)
						for (int j = 0; j < width; j++) {
							value = data[i * width + j + z * width * height];
							if (value < minLimit) {
								data[i * width + j + z * width * height] = 0;
							}
							else if (value > maxLimit) {
								data[i * width + j + z * width * height] = clustNum - 1;
							}
							else {
								int min = metric(clusters[0], value);
								int argmin = 0;
								for (int l = 1; l < clustNum; l++) {
									int current = metric(clusters[l], value);
									if (current < min) {
										min = current;
										argmin = l;
									}
								}

								sum[argmin] += value;
								number[argmin]++;
							}
						}

				for (int i = 0; i < clustNum; i++) {
					if (number[i] != 0) {
						uchar newValue = (uchar)(sum[i] / number[i]);
						clusters[i] = newValue;
					}
					number[i] = sum[i] = 0;
				}

				iter++;
			}

			delete[] sum, number;

			for (int z = 0; z < deep; z++)
				for (int i = 0; i < height; i++)
					for (int j = 0; j < width; j++) {
						value = data[i * width + j + z * width * height];
						if (value >= minLimit || value <= maxLimit) {
							int err = 0;
							int minErr = metric(value, clusters[0]);
							data[i * width + j + z * width * height] = 0;
							for (int l = 1; l < clustNum; l++) {
								err = metric(value, clusters[l]);
								if (err < minErr) {
									minErr = err;
									data[i * width + j + z * width * height] = l;
								}
							}
						}
					}
		}

		void quantizationByClusters(uchar* data, int width, int height, int deep, int clustNum, uchar* clusters) {
			for (int z = 0; z < deep; z++)
				for (int i = 0; i < height; i++)
					for (int j = 0; j < width; j++) {
						uchar value = data[i * width + j + z * width * height];
						int err = 0;
						int minErr = metric(value, clusters[0]);
						data[i * width + j + z * width * height] = 0;
						for (int l = 1; l < clustNum; l++) {
							err = metric(value, clusters[l]);
							if (err < minErr) {
								minErr = err;
								data[i * width + j + z * width * height] = l;
							}
						}
					}
		}

		void quantizationGreyLevel(uchar* data, int width, int height, int deep, int clustNum, System::String^ methodType) {
			if(methodType->Equals("Uniform")) {
				uniformQuantization(data, width, height, deep, clustNum);
			}
			else if(methodType->Equals("K-Means")) {
				kMeansQuantization(data, width, height, deep, clustNum);
			}
		}

		void glcmHor(uchar* data, int width, int height, int deep, int clustNum, float* glcm) {
			for (int i = 0; i < clustNum * clustNum; i++) glcm[i] = 0;
			int z = deep;
			for (int i = 0; i < height; i++)
					for (int j = 0; j < width - 1; j++)
						{
							int cur = (int)data[i * width + j + z * width * height];
							int next = (int)data[i * width + j + z * width * height + 1];
							glcm[next * clustNum + cur]++;
							glcm[cur * clustNum + next]++;
							}
		}

		void glcmCommonHor(uchar* data, int width, int height, int deep, int clustNum, float* glcm) {
			for (int i = 0; i < clustNum * clustNum; i++) glcm[i] = 0;
			int z = deep;
			for (int i = 0; i < height; i++)
				for (int j = 0; j < width - 1; j++)
					if (data[i * width + j + z * width * height] != 255 && data[i * width + j + z * width * height + 1] != 255)
					{
						int cur = (int)data[i * width + j + z * width * height];
						int next = (int)data[i * width + j + z * width * height + 1];
						glcm[next * clustNum + cur]++;
						glcm[cur * clustNum + next]++;
					}
		}

		void glcmVer(uchar* data, int width, int height, int deep, int clustNum, float* glcm) {
			for (int i = 0; i < clustNum * clustNum; i++) glcm[i] = 0;
			int z = deep;
				for (int i = 0; i < height-1; i++)
					for (int j = 0; j < width; j++)
						{
							int cur = (int)data[i * width + j + z * width * height];
							int next = (int)data[(i + 1) * width + j + z * width * height];
							glcm[next * clustNum + cur]++;
							glcm[cur * clustNum + next]++;
						}
		}

		void glcm45(uchar* data, int width, int height, int deep, int clustNum, float* glcm) {
			for (int i = 0; i < clustNum * clustNum; i++) glcm[i] = 0;
			int z = deep;
			for (int i = 1; i < height; i++)
				for (int j = 0; j < width - 1; j++)
					{
						int cur = (int)data[i * width + j + z * width * height];
						int next = (int)data[(i - 1) * width + j + 1 + z * width * height];
						glcm[next * clustNum + cur]++;
						glcm[cur * clustNum + next]++;
					}
		}

		void glcm135(uchar* data, int width, int height, int deep, int clustNum, float* glcm) {
			for (int i = 0; i < clustNum * clustNum; i++) glcm[i] = 0;
			int z = deep;
			for (int i = 0; i < height - 1; i++)
				for (int j = 0; j < width - 1; j++)
					{
						int cur = (int)data[i * width + j + z * width * height];
						int next = (int)data[(i + 1) * width + j + 1 + z * width * height];
						glcm[next * clustNum + cur]++;
						glcm[cur * clustNum + next]++;
					}
		}

		void glcmNormalize(float* glcm, int levelNum) {
			double sum = 0;
			for (int i = 0; i < levelNum; i++)
				for (int j = 0; j < levelNum; j++) {
					sum += glcm[i * levelNum + j];
				}
			if (sum != 0) {
				for (int i = 0; i < levelNum; i++)
					for (int j = 0; j < levelNum; j++) {
						glcm[i * levelNum + j] /= sum;
					}
			}
		}

		void getMean(float* glcm, int levelNum, statistics& stat) {
			stat.meanX = stat.meanY = 0;
			for (int i = 0; i < levelNum; i++)
				for (int j = 0; j < levelNum; j++) {
					stat.meanX += i * glcm[i * levelNum + j];
					stat.meanY += j * glcm[i * levelNum + j];
				}
		}

		void getDeviation(float* glcm, int levelNum, statistics& stat) {
			stat.deviationX = stat.deviationY = 0;
			getMean(glcm, levelNum, stat);
			for (int i = 0; i < levelNum; i++)
				for (int j = 0; j < levelNum; j++) {
					stat.deviationX += glcm[i * levelNum + j] * (i - stat.meanX) * (i - stat.meanX);
					stat.deviationY += glcm[i * levelNum + j] * (j - stat.meanY) * (j - stat.meanY);
				}
			stat.deviationX = sqrt(stat.deviationX);
			stat.deviationY = sqrt(stat.deviationY);
		}

		double getEnergy(float* glcm, int levelNum) {
			double energy = 0;
			for (int i = 0; i < levelNum; i++)
				for (int j = 0; j < levelNum; j++) {
					energy += glcm[i * levelNum + j] * glcm[i * levelNum + j];
				}
			return energy;
		}

		double getContrast(float* glcm, int levelNum) {
			double contrast = 0;
			for (int i = 0; i < levelNum; i++)
				for (int j = 0; j < levelNum; j++) {
					contrast += (i - j) * (i - j) * glcm[i * levelNum + j];
				}
			return contrast;
		}

		double getHomogeneity(float* glcm, int levelNum) {
			double homogeneity = 0;
			for (int i = 0; i < levelNum; i++)
				for (int j = 0; j < levelNum; j++) {
					homogeneity += 1 / (1 + (i - j) * (i - j)) * glcm[i * levelNum + j];
				}
			return homogeneity;
		}

		double getCorrelation(float* glcm, int levelNum) {
			double correlation = 0;
			statistics stat;
			getDeviation(glcm, levelNum, stat);
			if (stat.deviationX != 0 && stat.deviationY != 0) {
				for (int i = 0; i < levelNum; i++)
					for (int j = 0; j < levelNum; j++) {
						correlation += (i - stat.meanX) * (j - stat.meanY) * glcm[i * levelNum + j];
					}
				correlation /= stat.deviationX * stat.deviationY;
			}
			return correlation;
		}

		double getEntropy(float* glcm, int levelNum) {
			double entropy = 0;
			for (int i = 0; i < levelNum; i++)
				for (int j = 0; j < levelNum; j++) {
					if (glcm[i * levelNum + j] > 0) entropy += -glcm[i * levelNum + j] * log(glcm[i * levelNum + j]);
				}
			return entropy;
		}

		double getVariance(float* glcm, int levelNum) {
			double result = 0;
			statistics stat;
			getMean(glcm, levelNum, stat);
			for (int i = 0; i < levelNum; i++)
				for (int j = 0; j < levelNum; j++) {
					result += (i - stat.meanX) * (i - stat.meanX) * glcm[i * levelNum + j];
				}
			return result;
		}

		double getSumAverage(float* glcm, int levelNum) {
			double result = 0;
			double p;
			int j;
			for (int k = 0; k <= 2 * (levelNum-1); k++) {
				p = 0;
				for (int i = 0; i <= k && i < levelNum && k - i < levelNum; i++) {
					j = k - i;
					p += glcm[i * levelNum + j];
				}
				result += (k+1) * p;
			}
			return result;
		}

		double getSumEntropy(float* glcm, int levelNum) {
			double entropy = 0;
			double p;
			int j;
			for (int k = 0; k <= 2 * (levelNum - 1); k++) {
				p = 0;
				for (int i = 0; i <= k && i < levelNum && k - i < levelNum; i++) {
					j = k - i;
					p += glcm[i * levelNum + j];
				}
				if (p > 0) entropy += -p * log(p);
			}
			return entropy;
		}

		double getSumVariance(float* glcm, int levelNum, double sumEntropy) {
			double result = 0;
			double p;
			int j;
			for (int k = 0; k <= 2 * (levelNum - 1); k++) {
				p = 0;
				for (int i = 0; i <= k && i < levelNum && k - i < levelNum; i++) {
					j = k - i;
					p += glcm[i * levelNum + j];
				}
				result += (k - sumEntropy) * (k - sumEntropy) * p;
			}
			return result;
		}

		double getDifferenceEntropy(float* glcm, int levelNum) {
			double entropy = 0;
			double p;
			int j;
			for (int k = 0; k < levelNum; k++) {
				p = 0;
				for (int i = 0; i < levelNum; i++) {
					j = (i < k) ? k - i : i - k;
					p += glcm[i * levelNum + j];
				}
				if (p > 0) entropy += -p * log(p);
			}
			return entropy;
		}

		double getDifferenceVariance(float* glcm, int levelNum, double diffEntropy) {
			double result = 0;
			double p;
			int j;
			for (int k = 0; k < levelNum; k++) {
				p = 0;
				for (int i = 0; i < levelNum; i++) {
					j = (i < k) ? k - i : i - k;
					p += glcm[i * levelNum + j];
				}
				result += (k - diffEntropy) * (k - diffEntropy) * p;
			}
			return result;
		}

		void binarization(uchar* data, int T) {
			Mat img = Mat(HEIGHT, WIDTH, CV_8UC1, data);

			for (int i = 0; i < img.rows; i++)
				for (int j = 0; j < img.cols; j++) {
					if (img.at<uchar>(i, j) < T) img.at<uchar>(i, j) = 0;
					else img.at<uchar>(i, j) = 255;
				}
		}

		double getQuadError(double *vec1, double* vec2, int n) {
			double sqr = 0;
			for (int i = 0; i < n; i++) {
				sqr += (vec2[i] - vec1[i]) * (vec2[i] - vec1[i]);
			}
			return sqrt(sqr);
		}

		double getAbsError(double* vec1, double* vec2, int n) {
			double error = 0;
			for (int i = 0; i < n; i++) {
				error += abs(vec2[i] - vec1[i]);
			}
			return error;
		}

		double getMaxError(double* vec1, double* vec2, int n) {
			double error = 0;
			for (int i = 0; i < n; i++) {
				error = (error < abs(vec2[i] - vec1[i])) ? abs(vec2[i] - vec1[i]) : error;
			}
			return error;
		}

		double getError(double *vec1, double* vec2, int n) {
			if (comboBox1->Text == "ABS")
				return getAbsError(vec1, vec2, n);
			else if (comboBox1->Text == "QUAD")
				return getQuadError(vec1, vec2, n);
			else getMaxError(vec1, vec2, n);
		}

		void getScalaVec(double* vec, float* glcm, int numLevel) {
			for (int k = 0; k < numLevel - 1; k++) {
				vec[k] = vec[numLevel - 1 + k] = 0;
				for (int i = 0; i < numLevel; i++) {
					vec[k] += glcm[k * numLevel + i] * glcm[(k + 1) * numLevel + i];
					vec[numLevel - 1 + k] += glcm[k + i * numLevel] * glcm[k + 1 + i * numLevel];
				}
			}
		}

		float getScal(float* glcm1, float* glcm2, int numLevel, int z) {
			float res = 0;
			for (int k = 0; k < numLevel * numLevel; k++) {
				res += glcm1[k + numLevel*numLevel*z] * glcm2[k];
			}
			return res;
		}

		void getFeatureVec(double* vec, float* glcm, int numLevel, float* glcmVer, float* glcm45, float* glcm135) {
			int i = 0;
			//vec[i++] = getEnergy(glcm, numLevel);//10//10
			vec[i++] = getContrast(glcm, numLevel);//8//8
			//vec[i++] = getHomogeneity(glcm, numLevel);//11//
			vec[i++] = getCorrelation(glcm, numLevel); //+//7//7
			//vec[i++] = getEntropy(glcm, numLevel);//9//9
			vec[i++] = getVariance(glcm, numLevel);    //+-//5//5
			vec[i++] = getSumAverage(glcm, numLevel);  //+//1//1
			vec[i++] = getSumEntropy(glcm, numLevel);  //+//6//6
			vec[i++] = getSumVariance(glcm, numLevel, getSumEntropy(glcm, numLevel));//-+//3//3
			vec[i++] = getDifferenceEntropy(glcm, numLevel);//4//4
			vec[i++] = getDifferenceVariance(glcm, numLevel, getDifferenceEntropy(glcm, numLevel)); //-+//2//2
			/*
			//vec[i++] = getEnergy(glcmVer, numLevel);
			vec[i++] = getContrast(glcmVer, numLevel);
			//vec[i++] = getHomogeneity(glcmVer, numLevel);
			vec[i++] = getCorrelation(glcmVer, numLevel);
			//vec[i++] = getEntropy(glcmVer, numLevel);
			vec[i++] = getVariance(glcmVer, numLevel);
			vec[i++] = getSumAverage(glcmVer, numLevel);
			vec[i++] = getSumEntropy(glcmVer, numLevel);
			vec[i++] = getSumVariance(glcmVer, numLevel, getSumEntropy(glcmVer, numLevel));
			vec[i++] = getDifferenceEntropy(glcmVer, numLevel);
			vec[i++] = getDifferenceVariance(glcmVer, numLevel, getDifferenceEntropy(glcmVer, numLevel));
			*/
			//vec[i++] = getEnergy(glcm45, numLevel);
			//vec[i++] = getContrast(glcm45, numLevel);
			//vec[i++] = getHomogeneity(glcm45, numLevel);
			//vec[i++] = getCorrelation(glcm45, numLevel);
			//vec[i++] = getEntropy(glcm45, numLevel);
			//vec[i++] = getVariance(glcm45, numLevel);
			//vec[i++] = getSumAverage(glcm45, numLevel);
			//vec[i++] = getSumEntropy(glcm45, numLevel);
			//vec[i++] = getSumVariance(glcm45, numLevel, vec[29]);
			//vec[i++] = getDifferenceEntropy(glcm45, numLevel);
			//vec[i++] = getDifferenceVariance(glcm45, numLevel, getDifferenceEntropy(glcm45, numLevel));
			
			//vec[i++] = getEnergy(glcm135, numLevel);
			//vec[i++] = getContrast(glcm135, numLevel);
			//vec[i++] = getHomogeneity(glcm135, numLevel);
			//vec[i++] = getCorrelation(glcm135, numLevel);
			//vec[i++] = getEntropy(glcm135, numLevel);
			//vec[i++] = getVariance(glcm135, numLevel);
			//vec[i++] = getSumAverage(glcm135, numLevel);
			//vec[i++] = getSumEntropy(glcm135, numLevel);
			//vec[i++] = getSumVariance(glcm135, numLevel, vec[40]);
			//vec[i++] = getDifferenceEntropy(glcm135, numLevel);
			//vec[i++] = getDifferenceVariance(glcm135, numLevel, getDifferenceEntropy(glcm135, numLevel));
			
		}

		int calculate(uchar* data, int width, int height, int deep, double *vec, int n) {
			int BLOCK_SIZE = int(numericUpDown2->Value);
			uchar* tmp = new uchar[BLOCK_SIZE * BLOCK_SIZE];
			float* horMat = new float[LEVEL_NUM * LEVEL_NUM];
			float* verMat = new float[LEVEL_NUM * LEVEL_NUM];
			float* mat45 = new float[LEVEL_NUM * LEVEL_NUM];
			float* mat135 = new float[LEVEL_NUM * LEVEL_NUM];
			double* things = new double[n];

			uchar* region = new uchar[SIZE];
			for (int z = 0; z < deep; z++)
				for (int i = 0; i < height; i++)
					for (int j = 0; j < width; j++) {
						region[i * width + j + z * width * height] = data[i * width + j + z * width * height];
					}
			if (comboBox3->Text->ToString() == "K-Means") quantizationByClusters(region, width, height, deep, LEVEL_NUM, clusters);
			else quantizationGreyLevel(region, width, height, deep, LEVEL_NUM, "Uniform");

			for (int z = 0; z < deep; z++)
				for (int i = BLOCK_SIZE / 2; i < height - BLOCK_SIZE / 2; i++)
					for (int j = BLOCK_SIZE / 2; j < width - BLOCK_SIZE / 2; j++) {
						for (int k = 0; k < BLOCK_SIZE; k++)
							for (int m = 0; m < BLOCK_SIZE; m++) {
								tmp[k * BLOCK_SIZE + m] = region[(i - BLOCK_SIZE / 2 + k) * width + j + z * height * width - BLOCK_SIZE / 2 + m];
							}
						glcmHor(tmp, BLOCK_SIZE, BLOCK_SIZE, 0, LEVEL_NUM, horMat);
						glcmVer(tmp, BLOCK_SIZE, BLOCK_SIZE, 0, LEVEL_NUM, verMat);
						glcm45(tmp, BLOCK_SIZE, BLOCK_SIZE, 0, LEVEL_NUM, mat45);
						glcm135(tmp, BLOCK_SIZE, BLOCK_SIZE, 0, LEVEL_NUM, mat135);
						glcmNormalize(horMat, LEVEL_NUM);
						glcmNormalize(verMat, LEVEL_NUM);
						glcmNormalize(mat45, LEVEL_NUM);
						glcmNormalize(mat135, LEVEL_NUM);
						getFeatureVec(things, horMat, LEVEL_NUM, verMat, mat45, mat135);
						errors[i * width + j + z * width * height] = getError(vec, things, n);
					}
					
			delete[] tmp, horMat, things, verMat, region, mat135, mat45;
			return 0;
		}

		int calculateByScal(uchar* data, int width, int height, int deep, float* glcm1, float *glcm2, float* glcm3, float* glcm4, int n) {
			int BLOCK_SIZE = int(numericUpDown2->Value);
			uchar* tmp = new uchar[BLOCK_SIZE * BLOCK_SIZE];
			float* horMat = new float[LEVEL_NUM * LEVEL_NUM];
			float* verMat = new float[LEVEL_NUM * LEVEL_NUM];
			float* mat45 = new float[LEVEL_NUM * LEVEL_NUM];
			float* mat135 = new float[LEVEL_NUM * LEVEL_NUM];
			double* things = new double[n];

			uchar* region = new uchar[SIZE];
			for (int z = 0; z < deep; z++)
				for (int i = 0; i < height; i++)
					for (int j = 0; j < width; j++) {
						region[i * width + j + z * width * height] = data[i * width + j + z * width * height];
					}
			if (comboBox3->Text->ToString() == "K-Means") quantizationByClusters(region, width, height, deep, LEVEL_NUM, clusters);
			else quantizationGreyLevel(region, width, height, deep, LEVEL_NUM, "Uniform");

			for (int z = 0; z < deep; z++)
				for (int i = BLOCK_SIZE / 2; i < height - BLOCK_SIZE / 2; i++)
					for (int j = BLOCK_SIZE / 2; j < width - BLOCK_SIZE / 2; j++) {
						for (int k = 0; k < BLOCK_SIZE; k++)
							for (int m = 0; m < BLOCK_SIZE; m++) {
								tmp[k * BLOCK_SIZE + m] = region[(i - BLOCK_SIZE / 2 + k) * width + j + z * height * width - BLOCK_SIZE / 2 + m];
							}
						glcmHor(tmp, BLOCK_SIZE, BLOCK_SIZE, 0, LEVEL_NUM, horMat);
						glcmVer(tmp, BLOCK_SIZE, BLOCK_SIZE, 0, LEVEL_NUM, verMat);
						glcm45(tmp, BLOCK_SIZE, BLOCK_SIZE, 0, LEVEL_NUM, mat45);
						glcm135(tmp, BLOCK_SIZE, BLOCK_SIZE, 0, LEVEL_NUM, mat135);

						glcmNormalize(horMat, LEVEL_NUM);
						glcmNormalize(verMat, LEVEL_NUM);
						glcmNormalize(mat45, LEVEL_NUM);
						glcmNormalize(mat135, LEVEL_NUM);

						errors[i * width + j + z * width * height] = 0;
						for (int x = 0; x < DEEP; x++) {
							errors[i * width + j + z * width * height] += getScal(glcm1, horMat, LEVEL_NUM, x);
							errors[i * width + j + z * width * height] += getScal(glcm2, verMat, LEVEL_NUM, x);
							errors[i * width + j + z * width * height] += getScal(glcm3, verMat, LEVEL_NUM, x);
							errors[i * width + j + z * width * height] += getScal(glcm4, verMat, LEVEL_NUM, x);
						}
						errors[i * width + j + z * width * height] /= DEEP * 4;
					}

			delete[] tmp, horMat, things, verMat, region, mat135, mat45;
			return 0;
		}


		void calculateByScal3(float* standard, uchar* testData, Mat record) {
			int BLOCK_SIZE = int(numericUpDown2->Value);
			uchar* tmp = new uchar[BLOCK_SIZE * BLOCK_SIZE];
			float* horMat = new float[LEVEL_NUM * LEVEL_NUM];
			float* verMat = new float[LEVEL_NUM * LEVEL_NUM];
			float* mat45 = new float[LEVEL_NUM * LEVEL_NUM];
			float* mat135 = new float[LEVEL_NUM * LEVEL_NUM];

			uchar* region = new uchar[HEIGHT* WIDTH];

			for (int i = 0; i < HEIGHT; i++)
					for (int j = 0; j < WIDTH; j++) {
						region[i * WIDTH + j] = testData[i * WIDTH + j];
					}

			if (comboBox3->Text->ToString() == "K-Means") quantizationByClusters(region, WIDTH, HEIGHT, DEEP, LEVEL_NUM, clusters);
			else quantizationGreyLevel(region, WIDTH, HEIGHT, DEEP, LEVEL_NUM, "Uniform");

			for (int i = BLOCK_SIZE / 2; i < HEIGHT - BLOCK_SIZE / 2; i++)
					for (int j = BLOCK_SIZE / 2; j < WIDTH - BLOCK_SIZE / 2; j++) {
						for (int k = 0; k < BLOCK_SIZE; k++)
							for (int m = 0; m < BLOCK_SIZE; m++) {
								tmp[k * BLOCK_SIZE + m] = region[(i - BLOCK_SIZE / 2 + k) * WIDTH + j- BLOCK_SIZE / 2 + m];
							}
						glcmHor(tmp, BLOCK_SIZE, BLOCK_SIZE, 0, LEVEL_NUM, horMat);
						glcmNormalize(horMat, LEVEL_NUM);

						record.at<double>(i, j) = getScal(standard, horMat, LEVEL_NUM, 0);
					}

			delete[] tmp, horMat, verMat, region, mat135, mat45;
		}

		int calculateByScal2(uchar* data, int width, int height, int deep, float* glcm1, float* glcm2, int n) {
			int BLOCK_SIZE = int(numericUpDown2->Value);
			uchar* tmp = new uchar[BLOCK_SIZE * BLOCK_SIZE];
			float* horMat = new float[LEVEL_NUM * LEVEL_NUM];
			float* verMat = new float[LEVEL_NUM * LEVEL_NUM];
			float* mat45 = new float[LEVEL_NUM * LEVEL_NUM];
			float* mat135 = new float[LEVEL_NUM * LEVEL_NUM];
			double* things = new double[n];

			uchar* region = new uchar[SIZE];
			for (int z = 0; z < deep; z++)
				for (int i = 0; i < height; i++)
					for (int j = 0; j < width; j++) {
						region[i * width + j + z * width * height] = data[i * width + j + z * width * height];
					}
			if (comboBox3->Text->ToString() == "K-Means") quantizationByClusters(region, width, height, deep, LEVEL_NUM, clusters);
			else quantizationGreyLevel(region, width, height, deep, LEVEL_NUM, "Uniform");

			for (int z = 0; z < deep; z++)
				for (int i = BLOCK_SIZE / 2; i < height - BLOCK_SIZE / 2; i++)
					for (int j = BLOCK_SIZE / 2; j < width - BLOCK_SIZE / 2; j++) {
						for (int k = 0; k < BLOCK_SIZE; k++)
							for (int m = 0; m < BLOCK_SIZE; m++) {
								tmp[k * BLOCK_SIZE + m] = region[(i - BLOCK_SIZE / 2 + k) * width + j + z * height * width - BLOCK_SIZE / 2 + m];
							}
						glcmHor(tmp, BLOCK_SIZE, BLOCK_SIZE, 0, LEVEL_NUM, horMat);
						glcmVer(tmp, BLOCK_SIZE, BLOCK_SIZE, 0, LEVEL_NUM, verMat);
						//glcm45(tmp, BLOCK_SIZE, BLOCK_SIZE, 0, LEVEL_NUM, mat45);
						//glcm135(tmp, BLOCK_SIZE, BLOCK_SIZE, 0, LEVEL_NUM, mat135);
						glcmNormalize(horMat, LEVEL_NUM);
						glcmNormalize(verMat, LEVEL_NUM);
						//glcmNormalize(mat45, LEVEL_NUM);
						//glcmNormalize(mat135, LEVEL_NUM);
						errors[i * width + j + z * width * height] = 0;

						errors[i * width + j + z * width * height] += getScal(glcm1, horMat, LEVEL_NUM, 0)/4;
						errors[i * width + j + z * width * height] += getScal(glcm2, verMat, LEVEL_NUM, 0)/4;

						errors[i * width + j + z * width * height] += getScal(glcm1, horMat, LEVEL_NUM, 1)/2;
						errors[i * width + j + z * width * height] += getScal(glcm2, verMat, LEVEL_NUM, 1)/2;

						errors[i * width + j + z * width * height] += getScal(glcm1, horMat, LEVEL_NUM, 2)/4;
						errors[i * width + j + z * width * height] += getScal(glcm2, verMat, LEVEL_NUM, 2)/4;

						errors[i * width + j + z * width * height] /= 2;
					}

			delete[] tmp, horMat, things, verMat, region, mat135, mat45;
			return 0;
		}

		void calculateByHist(uchar* data, int width, int height, int deep, double* etalon) {
			int BLOCK_SIZE = int(numericUpDown2->Value);
			int level_num = (int)numericUpDown3->Value;
			double* numbers = new double[level_num];

			uchar* region = new uchar[width * height * deep];
			for (int z = 0; z < deep; z++)
				for (int i = 0; i < height; i++)
					for (int j = 0; j < width; j++) {
						region[i * width + j + z * width * height] = data[i * width + j + z * width * height];
					}
			
			if (level_num != 256 && comboBox3->Text->ToString() == "K-Means") quantizationByClusters(region, width, height, deep, level_num, clusters);

			for (int z = 0; z < deep; z++)
				for (int i = BLOCK_SIZE / 2; i < height - BLOCK_SIZE / 2; i++)
					for (int j = BLOCK_SIZE / 2; j < width - BLOCK_SIZE / 2; j++) {
						for (int w = 0; w < level_num; w++) numbers[w] = 0;
						for (int k = 0; k < BLOCK_SIZE; k++)
							for (int m = 0; m < BLOCK_SIZE; m++) {
								numbers[region[(i - BLOCK_SIZE / 2 + k) * width + j + z * height * width - BLOCK_SIZE / 2 + m]]++;
							}
						errors[i * width + j + z * width * height] = 0;
						for (int w = 0; w < level_num; w++) {
							numbers[w] /= BLOCK_SIZE * BLOCK_SIZE;
							errors[i * width + j + z * width * height] += numbers[w] * etalon[w];
						}
					}

			delete[] region, numbers;
		}

		void getGlcmFromRegions(matrix reg) {
			float* horMat = new float[DEEP * LEVEL_NUM * LEVEL_NUM];
			float* verMat = new float[DEEP * LEVEL_NUM * LEVEL_NUM];
			float* mat45 = new float[LEVEL_NUM * LEVEL_NUM];
			float* mat135 = new float[LEVEL_NUM * LEVEL_NUM];
			uchar* region = new uchar[SIZE];

			for (int z = 0; z < DEEP; z++)
				for (int i = 0; i < HEIGHT; i++)
					for (int j = 0; j < WIDTH; j++) {
						region[i * WIDTH + j + z * WIDTH * HEIGHT] = reg.mat[z].at<uchar>(i, j);
						trainMemblock[i * WIDTH + j] = reg.data[i * WIDTH + j + 0 * WIDTH * HEIGHT];
						memblock[i * WIDTH + j + 0 * WIDTH * HEIGHT] = reg.data[i * WIDTH + j + 0 * WIDTH * HEIGHT];
					}

			if (checkBox4->Checked) {
				Mat trainMat = Mat(HEIGHT, WIDTH, CV_8UC1, trainMemblock);
				namedWindow("trainMat", WINDOW_NORMAL);
				setMouseCallback("trainMat", onMouseCallbackPointer);
				imshow("trainMat", trainMat);
			}
			else {
				quantizationGreyLevel(region, WIDTH, HEIGHT, DEEP, LEVEL_NUM, comboBox3->Text->ToString());

				for (int z = 0; z < DEEP; z++) {
					glcmHor(region, WIDTH, HEIGHT, z, LEVEL_NUM, &horMat[z * LEVEL_NUM * LEVEL_NUM]);
					glcmVer(region, WIDTH, HEIGHT, z, LEVEL_NUM, &verMat[z * LEVEL_NUM * LEVEL_NUM]);
					glcm45(region, WIDTH, HEIGHT, z, LEVEL_NUM, mat45);
					glcm135(region, WIDTH, HEIGHT, z, LEVEL_NUM, mat135);

					glcmNormalize(&horMat[z * LEVEL_NUM * LEVEL_NUM], LEVEL_NUM);
					glcmNormalize(&verMat[z * LEVEL_NUM * LEVEL_NUM], LEVEL_NUM);
					glcmNormalize(mat45, LEVEL_NUM);
					glcmNormalize(mat135, LEVEL_NUM);
				}

				if (checkBox3->Checked) {
					calculateByScal(trainMemblock, WIDTH, HEIGHT, 1, horMat, verMat, mat45, mat135, NUM_FEATURES);
				}
				else {
					getFeatureVec(vec, horMat, LEVEL_NUM, verMat, mat45, mat135);
					calculate(memblock, WIDTH, HEIGHT, 1, vec, NUM_FEATURES);
				}
			}

			delete[] horMat, verMat, mat45, mat135;
		}

		void getGlcmFromRegions2(matrix reg, Mat mat) {
			float* horMat = new float[DEEP * LEVEL_NUM * LEVEL_NUM];
			float* verMat = new float[DEEP * LEVEL_NUM * LEVEL_NUM];
			float* mat45 = new float[LEVEL_NUM * LEVEL_NUM];
			float* mat135 = new float[LEVEL_NUM * LEVEL_NUM];
			uchar* region = new uchar[SIZE];

			for (int z = 0; z < DEEP; z++)
				for (int i = 0; i < HEIGHT; i++)
					for (int j = 0; j < WIDTH; j++) {
						region[i * WIDTH + j + z * WIDTH * HEIGHT] = reg.mat[z].at<uchar>(i, j);
					}

			quantizationGreyLevel(region, WIDTH, HEIGHT, DEEP, LEVEL_NUM, comboBox3->Text->ToString());

			for (int z = 0; z < DEEP; z++) {
				glcmHor(region, WIDTH, HEIGHT, z, LEVEL_NUM, &horMat[z * LEVEL_NUM * LEVEL_NUM]);
				glcmVer(region, WIDTH, HEIGHT, z, LEVEL_NUM, &verMat[z * LEVEL_NUM * LEVEL_NUM]);
				//glcm45(region, WIDTH, HEIGHT, DEEP, LEVEL_NUM, mat45);
				//glcm135(region, WIDTH, HEIGHT, DEEP, LEVEL_NUM, mat135);

				glcmNormalize(&horMat[z * LEVEL_NUM * LEVEL_NUM], LEVEL_NUM);
				glcmNormalize(&verMat[z * LEVEL_NUM * LEVEL_NUM], LEVEL_NUM);

				//glcmNormalize(mat45, LEVEL_NUM);
				//glcmNormalize(mat135, LEVEL_NUM);

				getFeatureVec(vec, &horMat[z * LEVEL_NUM * LEVEL_NUM], LEVEL_NUM, &verMat[z * LEVEL_NUM * LEVEL_NUM], mat45, mat135);

				for (int i = 0; i < NUM_FEATURES; i++) {
					mat.at<double>(z, i) = vec[i];
				}
			}

			delete[] horMat, verMat, mat45, mat135;
		}

		void equalizeHist(uchar* data, int width, int height, int deep) {
			int* hist = new int[256];
			int min = INT8_MAX;
			int* cdf = new int[256];

			for (int i = 0; i < 256; i++) {
				hist[i] = 0;
			}

			for (int z = 0; z < deep; z++)
				for (int i = 0; i < height; i++)
					for (int j = 0; j < width; j++) {
						int x = data[i * width + j + z * width * height];
						hist[x]++;
					}

			cdf[0] = hist[0];
			min = (cdf[0] == 0) ? min : cdf[0];
			for (int i = 1; i < 256; i++) {
				cdf[i] = cdf[i - 1] + hist[i];
				if (cdf[i] != 0 && cdf[i] < min) {
					min = cdf[i];
				}
			}

			for (int z = 0; z < deep; z++)
				for (int i = 0; i < height; i++)
					for (int j = 0; j < width; j++) {
						int x = data[i * width + j + z * width * height];
						if (cdf[x] == 0) cdf[x] = min;
						data[i * width + j + z * width * height] = (uchar)(((float)(cdf[x] - min)) / (width * height * deep - 1) * 255);
					}
			delete[] hist, cdf;
		}
		
		void writeCSV(string filename, Mat m)
		{
			ofstream myfile;
			myfile.open(filename.c_str(), ios::out | ios::app);
			myfile << cv::format(m, cv::Formatter::FMT_CSV) << std::endl;
			myfile.close();
		}

		void readCSV(string filename)
		{
			ifstream stream(filename);
			vector<uchar> data;

			string line;
			int index = 0;
			while (getline(stream, line)) {
				istringstream s(line);
				string field;
				while (getline(s, field, ',')) {
					data.push_back(stoi(field)==0?0:255);
				}
				Mat mat = Mat(cv::Size(WIDTH, HEIGHT), CV_8UC1, &data[0]);
				namedWindow(to_string(index));
				imshow(to_string(index), mat);
				imshow(to_string(index), mat);
				data.clear();
				index++;
			}
			waitKey(0);
		}

		void calc(uchar* data, Mat mat, int deep) {
			int blockSize = int(numericUpDown2->Value);
			uchar* blockData = new uchar[blockSize * blockSize];
			float* horGLCM = new float[LEVEL_NUM * LEVEL_NUM];
			float* verGLCM = new float[LEVEL_NUM * LEVEL_NUM];
			float* mat45 = new float[LEVEL_NUM * LEVEL_NUM];
			float* mat135 = new float[LEVEL_NUM * LEVEL_NUM];
			double* features = new double[NUM_FEATURES];
			for (int z = 0; z < deep; z++)
				for (int i = blockSize / 2; i < HEIGHT - blockSize / 2; i++)
					for (int j = blockSize / 2; j < WIDTH - blockSize / 2; j++) {
						for (int k = 0; k < blockSize; k++)
							for (int m = 0; m < blockSize; m++) {
								blockData[k * blockSize + m] = data[(i - blockSize / 2 + k) * WIDTH + j + z * HEIGHT * WIDTH - blockSize / 2 + m];
							}
						glcmHor(blockData, blockSize, blockSize, 0, LEVEL_NUM, horGLCM);
						glcmVer(blockData, blockSize, blockSize, 0, LEVEL_NUM, verGLCM);
						glcm45(blockData, blockSize, blockSize, 0, LEVEL_NUM, mat45);
						glcm135(blockData, blockSize, blockSize, 0, LEVEL_NUM, mat135);
						glcmNormalize(horGLCM, LEVEL_NUM);
						glcmNormalize(verGLCM, LEVEL_NUM);
						glcmNormalize(mat45, LEVEL_NUM);
						glcmNormalize(mat135, LEVEL_NUM);
						getFeatureVec(features, horGLCM, LEVEL_NUM, verGLCM, mat45, mat135);
						for (int l = 0; l < NUM_FEATURES; l++) {
							mat.at<double>(z * WIDTH * HEIGHT + i * WIDTH + j, l) = features[l];
						}
					}
			delete[] blockData, horGLCM, features, mat45, mat135, verGLCM;
		}

		void getHistParams(int* hist, int size, double* params) {
			double sum = 0, aver = 0, sigma = 0, min = 255, max = 0, median = 0;
			int argmin = 0, argmax = 0, count = 0;
			for (int i = 0; i < size; i++) {
				if (hist[i] != 0)
					count++;
				sum += hist[i];
			}
			aver = sum / count;

			sum = 0;
			double current = 0;

			for (int i = 0; i < size; i++) {
				current = abs(hist[i] - aver);
				if (hist[i] != 0 && hist[i] < min) {
					min = hist[i];
					argmin = i;
				}
				 
				if (hist[i] > max) {
					max = hist[i];
					argmax = i;
				}

				sum += current * current;
			}
			sigma = sqrt(sum / size);

			sort(hist, hist + size);

			int i = 0;
			for (i = 0; i < size; i++) 
				if (hist[i] != 0) break;

			median = (hist[(size + i) / 2] + hist[(size - 1 + i) / 2]) / 2;

			params[0] = argmin;
			params[1] = argmax;
			params[2] = aver;
			params[3] = sigma;
			params[4] = median;
			params[5] = (median - aver) / sigma;
		}

		void extractFeatures() {
			DEEP = (int)numericUpDown6->Value;
			LEVEL_NUM = (int)numericUpDown3->Value;
			SIZE = WIDTH * HEIGHT * DEEP;
			int pos = 0;
			int HIST_PARAMS = 6;

			clusters = new uchar[LEVEL_NUM];

			int blockSize = int(numericUpDown2->Value);
			FILE* trainingFile = fopen("Data/training.raw", "rb");
			FILE* markingTrainingFile = fopen("Data/training_groundtruth.raw", "rb");

			FILE* testingFile = fopen("Data/testing.raw", "rb");
			FILE* markingTestingFile = fopen("Data/testing_groundtruth.raw", "rb");

			vector<double> trainVec;
			
			int step = 165.0f / DEEP;
			int mark;

			uchar* data = new uchar[SIZE];
			uchar* marks = new uchar[SIZE];

			for (int i = 0; i < DEEP; i++) {
				fseek(trainingFile, HEIGHT * WIDTH * i * step, SEEK_SET);
				fread(&data[HEIGHT * WIDTH * i], sizeof(uchar), HEIGHT * WIDTH, trainingFile);

				fseek(markingTrainingFile, HEIGHT * WIDTH * i * step, SEEK_SET);
				fread(&marks[HEIGHT * WIDTH * i], sizeof(uchar), HEIGHT * WIDTH, markingTrainingFile);
			}
			
			fclose(trainingFile);
			fclose(markingTrainingFile);

			Mat training = Mat(HEIGHT * WIDTH * DEEP, NUM_FEATURES + pos + 1 + LEVEL_NUM + HIST_PARAMS + 1, CV_64F);

			for (int z = 0; z < DEEP; z++)
				for (int i = blockSize / 2; i < HEIGHT - blockSize / 2; i++)
					for (int j = blockSize / 2; j < WIDTH - blockSize / 2; j++) {
						training.at<double>(z * WIDTH * HEIGHT + i * WIDTH + j, NUM_FEATURES + pos + LEVEL_NUM) =
							((double)data[z * WIDTH * HEIGHT + i * WIDTH + j])/255;
					}

			uchar* tmp = new uchar[blockSize * blockSize];
			int* numbers = new int[256];
			double* params = new double[HIST_PARAMS];

			for (int z = 0; z < DEEP; z++)
				for (int i = blockSize / 2; i < HEIGHT - blockSize / 2; i++)
					for (int j = blockSize / 2; j < WIDTH - blockSize / 2; j++) {
						for (int k = 0; k < 256; k++) numbers[k] = 0;
						for (int k = 0; k < blockSize; k++)
							for (int m = 0; m < blockSize; m++) {
								tmp[k * blockSize + m] = data[z * WIDTH * HEIGHT + (i - blockSize / 2 + k) * WIDTH + j - blockSize / 2 + m];
								numbers[tmp[k * blockSize + m]]++;
							}
						getHistParams(numbers, 256, params);
						for (int k = 0; k < HIST_PARAMS; k++) {
							training.at<double>(z * WIDTH * HEIGHT + i * WIDTH + j, NUM_FEATURES + pos + LEVEL_NUM + 1 + k) = params[k];
						}
					}

			quantizationGreyLevel(data, WIDTH, HEIGHT, DEEP, LEVEL_NUM, comboBox3->Text->ToString());
			
			delete[] numbers;
			numbers = new int[LEVEL_NUM];

			int firstClassCount = 0;
			
			for (int z = 0; z < DEEP; z++)
				for (int i = blockSize / 2; i < HEIGHT - blockSize / 2; i++)
					for (int j = blockSize / 2; j < WIDTH - blockSize / 2; j++) {
						firstClassCount = 0;
						for (int k = 0; k < LEVEL_NUM; k++) numbers[k] = 0;
						for (int k = 0; k < blockSize; k++)
							for (int m = 0; m < blockSize; m++) {
								mark = marks[z * WIDTH * HEIGHT + (i - blockSize / 2 + k) * WIDTH + j - blockSize / 2 + m];
								if (mark != 0)
									firstClassCount++;
								tmp[k * blockSize + m] = data[z * WIDTH * HEIGHT + (i - blockSize / 2 + k) * WIDTH + j - blockSize / 2 + m];
								numbers[tmp[k * blockSize + m]]++;
							}
						for (int k = 0; k < LEVEL_NUM; k++) {
							training.at<double>(z * WIDTH * HEIGHT + i * WIDTH + j, NUM_FEATURES + pos + k) = ((double)numbers[k]);
						}

						mark = marks[z * WIDTH * HEIGHT + i * WIDTH + j];
						if (mark != 0) {
							if (firstClassCount == blockSize * blockSize)
								training.at<double>(z * WIDTH * HEIGHT + i * WIDTH + j, NUM_FEATURES + pos + 1 + LEVEL_NUM + HIST_PARAMS) = 1;
							else if (firstClassCount > blockSize* blockSize * 0.75)
								training.at<double>(z * WIDTH * HEIGHT + i * WIDTH + j, NUM_FEATURES + pos + 1 + LEVEL_NUM + HIST_PARAMS) = 1;
							else if (firstClassCount > blockSize* blockSize * 0.5)
								training.at<double>(z * WIDTH * HEIGHT + i * WIDTH + j, NUM_FEATURES + pos + 1 + LEVEL_NUM + HIST_PARAMS) = 0;
							else if (firstClassCount > blockSize* blockSize * 0.25)
								training.at<double>(z * WIDTH * HEIGHT + i * WIDTH + j, NUM_FEATURES + pos + 1 + LEVEL_NUM + HIST_PARAMS) = 0;
						}
						else {
							training.at<double>(z * WIDTH * HEIGHT + i * WIDTH + j, NUM_FEATURES + pos + 1 + LEVEL_NUM + HIST_PARAMS) = 0;
						}
					}
			

			for (int i = 0; i < LEVEL_NUM; i++)
				textBox1->AppendText(clusters[i] + "; ");

			calc(data, training, DEEP);

			/*
			addHeterogeneity(data, DEEP, 1, 3, training, NUM_FEATURES);
			addHeterogeneity(data, DEEP, 3, 5, training, NUM_FEATURES + 1);
			addHeterogeneity(data, DEEP, 3, 7, training, NUM_FEATURES + 2);
			addHeterogeneity(data, DEEP, 5, 7, training, NUM_FEATURES + 3);
			*/
			
			delete[] marks, data;

			writeCSV("Data/train.csv", training);

			//Testing data

			data = new uchar[HEIGHT * WIDTH * 1];

			delete[] numbers;
			numbers = new int[256];

			fread(data, sizeof(uchar), HEIGHT * WIDTH * 1, testingFile);
			fclose(testingFile);

			Mat testing = Mat(HEIGHT * WIDTH * 1 , NUM_FEATURES + pos + 1 + LEVEL_NUM + HIST_PARAMS + 1, CV_64F);

			for (int z = 0; z < 1; z++)
				for (int i = blockSize / 2; i < HEIGHT - blockSize / 2; i++)
					for (int j = blockSize / 2; j < WIDTH - blockSize / 2; j++) {
						testing.at<double>(z * WIDTH * HEIGHT + i * WIDTH + j, NUM_FEATURES + pos + LEVEL_NUM) =
							((double)data[z * WIDTH * HEIGHT + i * WIDTH + j])/255;
					}

			for (int z = 0; z < 1; z++)
				for (int i = blockSize / 2; i < HEIGHT - blockSize / 2; i++)
					for (int j = blockSize / 2; j < WIDTH - blockSize / 2; j++) {
						for (int k = 0; k < 256; k++) numbers[k] = 0;
						for (int k = 0; k < blockSize; k++)
							for (int m = 0; m < blockSize; m++) {
								tmp[k * blockSize + m] = data[z * WIDTH * HEIGHT + (i - blockSize / 2 + k) * WIDTH + j - blockSize / 2 + m];
								numbers[tmp[k * blockSize + m]]++;
							}

						getHistParams(numbers, 256, params);
						//for (int w = 0; w < 4; w++)
							//params[w] /= 255;
						for (int k = 0; k < HIST_PARAMS; k++) {
							testing.at<double>(z * WIDTH * HEIGHT + i * WIDTH + j, NUM_FEATURES + pos + LEVEL_NUM + 1 + k) = params[k];
						}
					}

			if (comboBox3->Text->ToString() == "K-Means") quantizationByClusters(data, WIDTH, HEIGHT, 1, LEVEL_NUM, clusters);
			else quantizationGreyLevel(data, WIDTH, HEIGHT, 1, LEVEL_NUM, "Uniform");

			calc(data, testing, 1);
			/*
			addHeterogeneity(data, 1, 1, 3, testing, NUM_FEATURES);
			addHeterogeneity(data, 1, 3, 5, testing, NUM_FEATURES + 1);
			addHeterogeneity(data, 1, 3, 7, testing, NUM_FEATURES + 2);
			addHeterogeneity(data, 1, 5, 7, testing, NUM_FEATURES + 3);
			*/

			marks = new uchar[HEIGHT * WIDTH * 1];
			fread(marks, sizeof(uchar), HEIGHT * WIDTH * 1, markingTestingFile);
			fclose(markingTestingFile);

			delete[] numbers;
			numbers = new int[LEVEL_NUM];
			
			for (int z = 0; z < 1; z++)
				for (int i = blockSize / 2; i < HEIGHT - blockSize / 2; i++)
					for (int j = blockSize / 2; j < WIDTH - blockSize / 2; j++) {
						firstClassCount = 0;
						for (int k = 0; k < LEVEL_NUM; k++) numbers[k] = 0;
						for (int k = 0; k < blockSize; k++)
							for (int m = 0; m < blockSize; m++) {
								mark = marks[z * WIDTH * HEIGHT + (i - blockSize / 2 + k) * WIDTH + j - blockSize / 2 + m];
								if (mark != 0)
									firstClassCount++;

								tmp[k * blockSize + m] = data[z * WIDTH * HEIGHT + (i - blockSize / 2 + k) * WIDTH + j - blockSize / 2 + m];
								numbers[tmp[k * blockSize + m]]++;
							}
						for (int k = 0; k < LEVEL_NUM; k++) {
							testing.at<double>(z * WIDTH * HEIGHT + i * WIDTH + j, NUM_FEATURES + pos + k) = ((double)numbers[k]);
						}

						mark = marks[z * WIDTH * HEIGHT + i * WIDTH + j];
						if (mark != 0) {
							if (firstClassCount == blockSize * blockSize)
								testing.at<double>(z * WIDTH * HEIGHT + i * WIDTH + j, NUM_FEATURES + pos + 1 + LEVEL_NUM + HIST_PARAMS) = 1;
							else if (firstClassCount > blockSize* blockSize * 0.75)
								testing.at<double>(z * WIDTH * HEIGHT + i * WIDTH + j, NUM_FEATURES + pos + 1 + LEVEL_NUM + HIST_PARAMS) = 1;
							else if (firstClassCount > blockSize* blockSize * 0.5)
								testing.at<double>(z * WIDTH * HEIGHT + i * WIDTH + j, NUM_FEATURES + pos + 1 + LEVEL_NUM + HIST_PARAMS) = 0;
							else if (firstClassCount > blockSize* blockSize * 0.25)
								testing.at<double>(z * WIDTH * HEIGHT + i * WIDTH + j, NUM_FEATURES + pos + 1 + LEVEL_NUM + HIST_PARAMS) = 0;

							testing.at<double>(z * WIDTH * HEIGHT + i * WIDTH + j, NUM_FEATURES + pos + 1 + LEVEL_NUM + HIST_PARAMS) = 1;
						}
						else {
							testing.at<double>(z * WIDTH * HEIGHT + i * WIDTH + j, NUM_FEATURES + pos + 1 + LEVEL_NUM + HIST_PARAMS) = 0;
						}
					}

			delete[] marks, data, numbers, tmp, params;
			
			writeCSV("Data/test.csv", testing);
		}

		void testTool() {
			DEEP = (int)numericUpDown6->Value;
			LEVEL_NUM = (int)numericUpDown3->Value;
			SIZE = WIDTH * HEIGHT * DEEP;

			clusters = new uchar[LEVEL_NUM];

			FILE* trainingFile = fopen("Data/training.raw", "rb");
			FILE* markingTrainingFile = fopen("Data/training_groundtruth.raw", "rb");
			FILE* testingFile = fopen("Data/testing.raw", "rb");
			FILE* markingTestingFile = fopen("Data/testing_groundtruth.raw", "rb");

			int step = 1;
			uchar** data = new uchar * [DEEP];
			uchar** marks = new uchar * [DEEP];
			for (int z = 0; z < DEEP; z++) {
				data[z] = new uchar[HEIGHT * WIDTH];
				marks[z] = new uchar[HEIGHT * WIDTH];
			}

			for (int i = 0; i < DEEP; i++) {
				fseek(trainingFile, HEIGHT * WIDTH * step * i, SEEK_SET);
				fseek(markingTrainingFile, HEIGHT * WIDTH * step * i, SEEK_SET);

				fread(data[i], sizeof(uchar), HEIGHT * WIDTH, trainingFile);
				fread(marks[i], sizeof(uchar), HEIGHT * WIDTH, markingTrainingFile);
			}
			fread(testingMarks, sizeof(uchar), HEIGHT * WIDTH, markingTestingFile);

			fclose(trainingFile);
			fclose(markingTrainingFile);
			fclose(markingTestingFile);

			uchar* test = new uchar[HEIGHT * WIDTH];

			fread(test, sizeof(uchar), HEIGHT * WIDTH, testingFile);
			fclose(testingFile);

			if (checkBox2->Checked) {
				equalizeHist(&data[0][0], WIDTH, HEIGHT, DEEP);
				equalizeHist(test, WIDTH, HEIGHT, DEEP);
			}

			matrix result;
			result.mat = new Mat[DEEP];

			for (int z = 0; z < DEEP; z++) {
				Mat training = Mat(HEIGHT, WIDTH, CV_8UC1, data[z]);
				Mat mask = Mat(HEIGHT, WIDTH, CV_8UC1, marks[z]);
				training.copyTo(result.mat[z], mask);
			}

			result.data = test;

			namedWindow("res", WINDOW_NORMAL);
			createTrackbar("EPS", "res", &value, 10000, ChangeThresholdCallbackPointer);

			namedWindow("res", WINDOW_NORMAL);
			createTrackbar("EPS2", "res", &value, 10000, ChangeThresholdCallbackPointer2);

			getGlcmFromRegions(result);

			draw();

			delete[] data, marks, test;
		}

		void expansionRegions() {
			DEEP = (int)numericUpDown6->Value;
			LEVEL_NUM = (int)numericUpDown3->Value;
			SIZE = WIDTH * HEIGHT * DEEP;

			clusters = new uchar[LEVEL_NUM];

			FILE* trainingFile = fopen("Data/training.raw", "rb");
			FILE* markingTrainingFile = fopen("Data/training_groundtruth.raw", "rb");
			FILE* testingFile = fopen("Data/testing.raw", "rb");
			FILE* markingTestingFile = fopen("Data/testing_groundtruth.raw", "rb");

			int step = 1;
			uchar** data = new uchar * [DEEP];
			uchar** marks = new uchar * [DEEP];
			for (int z = 0; z < DEEP; z++) {
				data[z] = new uchar[HEIGHT * WIDTH];
				marks[z] = new uchar[HEIGHT * WIDTH];
			}

			for (int i = 0; i < DEEP; i++) {
				fseek(trainingFile, HEIGHT * WIDTH * step * i, SEEK_SET);
				fseek(markingTrainingFile, HEIGHT * WIDTH * step * i, SEEK_SET);

				fread(data[i], sizeof(uchar), HEIGHT * WIDTH, trainingFile);
				fread(marks[i], sizeof(uchar), HEIGHT * WIDTH, markingTrainingFile);
			}
			fread(testingMarks, sizeof(uchar), HEIGHT * WIDTH, markingTestingFile);

			fclose(trainingFile);
			fclose(markingTrainingFile);
			fclose(markingTestingFile);

			ifstream stream("Data/predict.csv");
			vector<uchar> predict;

			string line;
			Mat predictMat;
			while (getline(stream, line)) {
				istringstream s(line);
				string field;
				while (getline(s, field, ',')) {
					predict.push_back(stoi(field) == 0 ? 0 : 255);
				}
				predictMat = Mat(cv::Size(WIDTH, HEIGHT), CV_8UC1, &predict[0]);
			}

			uchar* test = new uchar[HEIGHT * WIDTH];

			fread(test, sizeof(uchar), HEIGHT * WIDTH, testingFile);
			fclose(testingFile);

			if (checkBox2->Checked) {
				equalizeHist(&data[0][0], WIDTH, HEIGHT, DEEP);
				equalizeHist(test, WIDTH, HEIGHT, DEEP);
			}

			matrix result;
			result.mat = new Mat[DEEP];

			for (int z = 0; z < DEEP; z++) {
				Mat training = Mat(HEIGHT, WIDTH, CV_8UC1, data[z]);
				Mat mask = Mat(HEIGHT, WIDTH, CV_8UC1, marks[z]);
				training.copyTo(result.mat[z], mask);
			}

			result.data = test;

			namedWindow("res", WINDOW_NORMAL);

			getGlcmFromRegions(result);

			for (int i = 0; i < HEIGHT; i++)
				for (int j = 0; j < WIDTH; j++) {
					if (predictMat.at<uchar>(i, j) == 255)
						memblock[i * WIDTH + j] = 255;
				}

			vector<pair<int, int>> points;

			for (int i = 0; i < HEIGHT; i++)
				for (int j = 0; j < WIDTH; j++) {
					if (memblock[i * WIDTH + j] == 255) {
						points.clear();
						points.push_back(make_pair(i, j));
						collectRegion(memblock, i, j, 0, points);
						if (points.size() < COUNT) {
							for each (pair<int, int> point in points) {
								memblock[point.first * WIDTH + point.second] = test[point.first * WIDTH + point.second];
							}
						}
					}
				}

			namedWindow("memblock", WINDOW_NORMAL);
			Mat res = Mat(HEIGHT, WIDTH, CV_8UC1, memblock);
			imshow("memblock", res);

			autoExpansionRegions();

			delete[] data, marks, test;
		}

		void draw() {
			uchar* data = new uchar[HEIGHT * WIDTH];
			double error = 0;
			for (int i = 0; i < HEIGHT; i++)
				for (int j = 0; j < WIDTH; j++) {
					if (checkBox3->Checked) {
						if (errors[i * WIDTH + j] > EPS) {
							data[i * WIDTH + j] = 255;
							error += (testingMarks[i * WIDTH + j] == 255) ? 0 : 1;
						}
						else {
							data[i * WIDTH + j] = memblock[i * WIDTH + j];
							error += (testingMarks[i * WIDTH + j] != 255) ? 0 : 1;
						}
					}
					else {
						if (errors[i * WIDTH + j] == 0) {
							data[i * WIDTH + j] = memblock[i * WIDTH + j];
						}
						else if (errors[i * WIDTH + j] <= EPS) {
							data[i * WIDTH + j] = 255;
							error += (testingMarks[i * WIDTH + j] == 255) ? 0 : 1;
						}
						else {
							data[i * WIDTH + j] = memblock[i * WIDTH + j];
							error += (testingMarks[i * WIDTH + j] != 255) ? 0 : 1;
						}
					}
				}
			textBox1->Text = "Error = " + error / (WIDTH * HEIGHT);
			Mat res = Mat(HEIGHT, WIDTH, CV_8UC1, data);
			imshow("res", res);
		}

		void fill(uchar* source, int i, int j, int z, float eps, bool mode, vector<pair<int, int>>& points, float count) {
			uchar nextValue;
			double nextError;

			if (count > 0 && points.size() > count)
				return;

			if (mode) {
				if (i - 1 >= 0) {
					nextValue = source[(i - 1) * WIDTH + j + z * WIDTH * HEIGHT];
					nextError = errors[(i - 1) * WIDTH + j];
					if (nextValue != 255 && nextError > eps) {
						source[(i - 1) * WIDTH + j + z * WIDTH * HEIGHT] = 255;
						points.push_back(make_pair(i - 1, j));
						fill(source, i - 1, j, z, eps, mode, points, count);
					}
				}
				if (i + 1 < HEIGHT) {
					nextValue = source[(i + 1) * WIDTH + j + z * WIDTH * HEIGHT];
					nextError = errors[(i + 1) * WIDTH + j];
					if (nextValue != 255 && nextError > eps) {
						source[(i + 1) * WIDTH + j + z * WIDTH * HEIGHT] = 255;
						points.push_back(make_pair(i + 1, j));
						fill(source, i + 1, j, z, eps, mode, points, count);
					}
				}
				if (j - 1 >= 0) {
					nextValue = source[i * WIDTH + j - 1 + z * WIDTH * HEIGHT];
					nextError = errors[i * WIDTH + j - 1];
					if (nextValue != 255 && nextError > eps) {
						source[i * WIDTH + j - 1 + z * WIDTH * HEIGHT] = 255;
						points.push_back(make_pair(i, j - 1));
						fill(source, i, j - 1, z, eps, mode, points, count);
					}
				}
				if (j + 1 < WIDTH) {
					nextValue = source[i * WIDTH + j + 1 + z * WIDTH * HEIGHT];
					nextError = errors[i * WIDTH + j + 1];
					if (nextValue != 255 && nextError > eps) {
						source[i * WIDTH + j + 1 + z * WIDTH * HEIGHT] = 255;
						points.push_back(make_pair(i, j + 1));
						fill(source, i, j + 1, z, eps, mode, points, count);
					}
				}
				if (i - 1 >= 0 && j - 1 >= 0) {
					nextValue = source[(i - 1) * WIDTH + j - 1 + z * WIDTH * HEIGHT];
					nextError = errors[(i - 1) * WIDTH + j - 1];
					if (nextValue != 255 && nextError > eps) {
						source[(i - 1) * WIDTH + j - 1 + z * WIDTH * HEIGHT] = 255;
						points.push_back(make_pair(i - 1, j - 1));
						fill(source, i - 1, j - 1, z, eps, mode, points, count);
					}
				}
				if (i - 1 >= 0 && j + 1 < WIDTH) {
					nextValue = source[(i - 1) * WIDTH + j + 1 + z * WIDTH * HEIGHT];
					nextError = errors[(i - 1) * WIDTH + j + 1];
					if (nextValue != 255 && nextError > eps) {
						source[(i - 1) * WIDTH + j + 1 + z * WIDTH * HEIGHT] = 255;
						points.push_back(make_pair(i - 1, j + 1));
						fill(source, i - 1, j + 1, z, eps, mode, points, count);
					}
				}
				if (i + 1 < HEIGHT && j + 1 < WIDTH) {
					nextValue = source[(i + 1) * WIDTH + j + 1 + z * WIDTH * HEIGHT];
					nextError = errors[(i + 1) * WIDTH + j + 1];
					if (nextValue != 255 && nextError > eps) {
						source[(i + 1) * WIDTH + j + 1 + z * WIDTH * HEIGHT] = 255;
						points.push_back(make_pair(i + 1, j + 1));
						fill(source, i + 1, j + 1, z, eps, mode, points, count);
					}
				}
				if (i + 1 < HEIGHT && j - 1 >= 0) {
					nextValue = source[(i + 1) * WIDTH + j - 1 + z * WIDTH * HEIGHT];
					nextError = errors[(i + 1) * WIDTH + j - 1];
					if (nextValue != 255 && nextError > eps) {
						source[(i + 1) * WIDTH + j - 1 + z * WIDTH * HEIGHT] = 255;
						points.push_back(make_pair(i + 1, j - 1));
						fill(source, i + 1, j - 1, z, eps, mode, points, count);
					}
				}
			}
			else {
				if (i - 1 >= 0) {
					nextValue = source[(i - 1) * WIDTH + j + z * WIDTH * HEIGHT];
					nextError = errors[(i - 1) * WIDTH + j];
					if (nextValue != 255 && nextError < eps) {
						source[(i - 1) * WIDTH + j + z * WIDTH * HEIGHT] = 255;
						points.push_back(make_pair(i - 1, j));
						fill(source, i - 1, j, z, eps, mode, points, count);
					}
				}
				if (i + 1 < HEIGHT) {
					nextValue = source[(i + 1) * WIDTH + j + z * WIDTH * HEIGHT];
					nextError = errors[(i + 1) * WIDTH + j];
					if (nextValue != 255 && nextError < eps) {
						source[(i + 1) * WIDTH + j + z * WIDTH * HEIGHT] = 255;
						points.push_back(make_pair(i + 1, j));
						fill(source, i + 1, j, z, eps, mode, points, count);
					}
				}
				if (j - 1 >= 0) {
					nextValue = source[i * WIDTH + j - 1 + z * WIDTH * HEIGHT];
					nextError = errors[i * WIDTH + j - 1];
					if (nextValue != 255 && nextError < eps) {
						source[i * WIDTH + j - 1 + z * WIDTH * HEIGHT] = 255;
						points.push_back(make_pair(i, j - 1));
						fill(source, i, j - 1, z, eps, mode, points, count);
					}
				}
				if (j + 1 < WIDTH) {
					nextValue = source[i * WIDTH + j + 1 + z * WIDTH * HEIGHT];
					nextError = errors[i * WIDTH + j + 1];
					if (nextValue != 255 && nextError < eps) {
						source[i * WIDTH + j + 1 + z * WIDTH * HEIGHT] = 255;
						points.push_back(make_pair(i, j + 1));
						fill(source, i, j + 1, z, eps, mode, points, count);
					}
				}
			}
		}

		void draw2() {
			uchar* data = new uchar[HEIGHT * WIDTH];
			vector<pair<int, int>> points;

			for (int i = 0; i < HEIGHT; i++)
				for (int j = 0; j < WIDTH; j++) {
					data[i * WIDTH + j] = memblock[i * WIDTH + j];
				}

			for (int i = 0; i < HEIGHT; i++)
				for (int j = 0; j < WIDTH; j++) {
					if (memblock[i * WIDTH + j] == 255) {
						points.clear();
						fill(data, i, j, 0, EPS2, checkBox3->Checked, points, -1);
					}
				}

			Mat res = Mat(HEIGHT, WIDTH, CV_8UC1, data);
			imshow("res", res);

			delete[] data;
		}

		void autoExpansionRegions() {
			float eps = 1;
			float step = 0.1;
			float percent = 0.5;
			float endStep = 0.000000001;
			bool isFail = false;

			uchar* data = new uchar[HEIGHT * WIDTH];

			for (int i = 0; i < HEIGHT; i++)
				for (int j = 0; j < WIDTH; j++) {
					data[i * WIDTH + j] = memblock[i * WIDTH + j];
				}

			vector<pair<int, int>> allPoints;
			vector<pair<int, int>> currentRegion;
			vector<pair<int, int>> addedPoints;
			
			int count = 0;
			
			for (int i = 0; i < HEIGHT; i++)
				for (int j = 0; j < WIDTH; j++) {
					if (memblock[i * WIDTH + j] == 255) {
						if (find(allPoints.begin(), allPoints.end(), make_pair(i, j)) == allPoints.end()) {
							currentRegion.clear();
							currentRegion.push_back(make_pair(i, j));
							collectRegion(memblock, i, j, 0, currentRegion);
							allPoints.insert(allPoints.end(), currentRegion.begin(), currentRegion.end());

							for each (pair<int, int> point in currentRegion) {
								eps = 1;
								step = 0.1;
								while (step > endStep) {
									isFail = false;

									addedPoints.clear();
									fill(data, point.first, point.second, 0, eps, checkBox3->Checked, addedPoints, currentRegion.size() * percent);

									if (addedPoints.size() > currentRegion.size()* percent) {
										for each (pair<int, int> newPoint in addedPoints) {
											data[newPoint.first * WIDTH + newPoint.second] = memblock[newPoint.first * WIDTH + newPoint.second];
										}
										isFail = true;
									}
									if (isFail) {
										eps += step;
										step /= 10;
									}
									else if (eps > step) {
										eps -= step;
									}
									else {
										step /= 10;
										eps -= step;
									}
								}
							}
						}
					}
				}

			Mat res = Mat(HEIGHT, WIDTH, CV_8UC1, data);
			imshow("res", res);

			Mat toRecord = Mat(res);
			for (int i = 0; i<HEIGHT;i++)
				for (int j = 0; j < WIDTH; j++) {
					toRecord.at<uchar>(i, j) = toRecord.at<uchar>(i, j) == 255 ? 1 : 0;
				}
			writeCSV("Data/expansion.csv", toRecord);

			currentRegion.clear();
			allPoints.clear();
			addedPoints.clear();
			delete[] data;
		}

		void collectRegion(uchar* source, int i, int j, int z, vector<pair<int, int>> &points) {
			uchar nextValue;

			if (points.size() > COUNT) return;

			if (i - 1 >= 0 && find(points.begin(), points.end(), make_pair(i - 1, j)) == points.end()) {
				nextValue = source[(i - 1) * WIDTH + j + z * WIDTH * HEIGHT];
				if (nextValue == 255) {
					points.push_back(make_pair(i - 1, j));
					collectRegion(source, i - 1, j, z, points);
				}
			}
			if (i + 1 < HEIGHT && find(points.begin(), points.end(), make_pair(i + 1, j)) == points.end()) {
				nextValue = source[(i + 1) * WIDTH + j + z * WIDTH * HEIGHT];
				if (nextValue == 255) {
					points.push_back(make_pair(i + 1, j));
					collectRegion(source, i + 1, j, z, points);
				}
			}
			if (j - 1 >= 0 && find(points.begin(), points.end(), make_pair(i, j - 1)) == points.end()) {
				nextValue = source[i * WIDTH + j - 1 + z * WIDTH * HEIGHT];
				if (nextValue == 255) {
					points.push_back(make_pair(i, j - 1));
					collectRegion(source, i, j - 1, z, points);
				}
			}
			if (j + 1 < WIDTH && find(points.begin(), points.end(), make_pair(i, j + 1)) == points.end()) {
				nextValue = source[i * WIDTH + j + 1 + z * WIDTH * HEIGHT];
				if (nextValue == 255) {
					points.push_back(make_pair(i, j + 1));
					collectRegion(source, i, j + 1, z, points);
				}
			}
		}

		void addHeterogeneity(uchar* source, int deep, int kernel, int blockSize, Mat data, int pos) {
			int kernelSum, sum;
			for (int z = 0; z < deep; z++)
				for (int i = blockSize / 2; i < HEIGHT - blockSize / 2; i++)
					for (int j = blockSize / 2; j < WIDTH - blockSize / 2; j++) {
						kernelSum = sum = 0;
						for (int k = 0; k < blockSize; k++)
							for (int m = 0; m < blockSize; m++) {
								if (k >= (blockSize - kernel) / 2 && k < (blockSize + kernel) / 2
									&& m >= (blockSize - kernel) / 2 && m < (blockSize + kernel) / 2) {
									kernelSum += (int)source[(i - blockSize / 2 + k) * WIDTH + j + z * HEIGHT * WIDTH - blockSize / 2 + m];
								}
								else {
									sum += (int)source[(i - blockSize / 2 + k) * WIDTH + j + z * HEIGHT * WIDTH - blockSize / 2 + m];
								}
							}
						if (sum != 0) data.at<double>(z * WIDTH * HEIGHT + i * WIDTH + j, pos) = ((float)kernelSum) / sum;
					}
		}

		void displayPredictData(System::String^ fileName) {
			char* fname = new char[fileName->Length + 1 + 5];
			fname[0] = Convert::ToChar("D");
			fname[1] = Convert::ToChar("a");
			fname[2] = Convert::ToChar("t");
			fname[3] = Convert::ToChar("a");
			fname[4] = Convert::ToChar("/");

			for (int i = 0; i < fileName->Length; i++)
			{
				fname[i + 5] = Convert::ToChar(fileName[i]);
			}

			fname[fileName->Length + 5] = 0;

			ifstream stream(fname);
			vector<double> data;

			string line;
			int index = 0;
			while (getline(stream, line)) {
				istringstream s(line);
				string field;
				while (getline(s, field, ',')) {
					data.push_back(stod(field));
				}
			}

			uchar* imgData = new uchar[HEIGHT * WIDTH];

			double max = data[0];

			double t;

			for (int i = 0; i < HEIGHT; i++)
				for (int j = 0; j < WIDTH; j++) {
					t = data[i * WIDTH + j];
					if (t > max) max = t;
				}

			for (int i = 0; i < HEIGHT; i++)
				for (int j = 0; j < WIDTH; j++) {
					t = data[i * WIDTH + j];
					imgData[i * WIDTH + j] = (int)(t/max * 255);
				}

			Mat mat = Mat(HEIGHT, WIDTH, CV_8UC1, imgData);
			namedWindow(fname, WINDOW_NORMAL);
			imshow(fname, mat);

			delete[] imgData;
		}

		void testKmeans() {
			DEEP = 1;
			LEVEL_NUM = (int)numericUpDown3->Value;
			SIZE = WIDTH * HEIGHT;

			clusters = new uchar[LEVEL_NUM];

			int blockSize = int(numericUpDown2->Value);
			FILE* trainingFile = fopen("Data/training.raw", "rb");
			FILE* markingTrainingFile = fopen("Data/training_groundtruth.raw", "rb");

			FILE* testingFile = fopen("Data/testing.raw", "rb");
			FILE* markingTestingFile = fopen("Data/testing_groundtruth.raw", "rb");

			int step = 165.0f / DEEP;
			int mark;

			uchar* data = new uchar[SIZE];
			uchar* marks = new uchar[SIZE];

			for (int i = 0; i < DEEP; i++) {
				fseek(trainingFile, HEIGHT * WIDTH * i * step, SEEK_SET);
				fread(&data[HEIGHT * WIDTH * i], sizeof(uchar), HEIGHT * WIDTH, trainingFile);

				fseek(markingTrainingFile, HEIGHT * WIDTH * i * step, SEEK_SET);
				fread(&marks[HEIGHT * WIDTH * i], sizeof(uchar), HEIGHT * WIDTH, markingTrainingFile);
			}

			quantizationGreyLevel(data, WIDTH, HEIGHT, DEEP, LEVEL_NUM, "K-Means");

			Mat trainMat = Mat(HEIGHT, WIDTH, CV_8UC1, data);
			namedWindow("trainMat", WINDOW_NORMAL);
			imshow("trainMat", trainMat);

			delete[] marks, data;

			//Testing data

			data = new uchar[HEIGHT * WIDTH * 1];
			fread(data, sizeof(uchar), HEIGHT * WIDTH * 1, testingFile);
			fclose(testingFile);

			for (int i = 0; i < LEVEL_NUM; i++)
				textBox1->AppendText(clusters[i] + "; ");

			if (comboBox3->Text->ToString() == "K-Means") quantizationByClusters(data, WIDTH, HEIGHT, DEEP, LEVEL_NUM, clusters);

			Mat testMat = Mat(HEIGHT, WIDTH, CV_8UC1, data);
			namedWindow("testMat", WINDOW_NORMAL);
			imshow("testMat", testMat);

			delete[] data, marks;
		}

		void displayHist() {
			DEEP = 1;
			LEVEL_NUM = (int)numericUpDown3->Value;
			SIZE = WIDTH * HEIGHT;

			clusters = new uchar[LEVEL_NUM];

			int blockSize = int(numericUpDown2->Value);
			FILE* trainingFile = fopen("Data/training.raw", "rb");
			FILE* markingTrainingFile = fopen("Data/training_groundtruth.raw", "rb");

			FILE* testingFile = fopen("Data/testing.raw", "rb");
			FILE* markingTestingFile = fopen("Data/testing_groundtruth.raw", "rb");

			int step = 165.0f / DEEP;
			int mark;

			uchar* data = new uchar[SIZE];
			uchar* marks = new uchar[SIZE];

			for (int i = 0; i < DEEP; i++) {
				fseek(trainingFile, HEIGHT * WIDTH * i * step, SEEK_SET);
				fread(&data[HEIGHT * WIDTH * i], sizeof(uchar), HEIGHT * WIDTH, trainingFile);

				fseek(markingTrainingFile, HEIGHT * WIDTH * i * step, SEEK_SET);
				fread(&marks[HEIGHT * WIDTH * i], sizeof(uchar), HEIGHT * WIDTH, markingTrainingFile);
			}

			Mat trainMat = Mat(HEIGHT, WIDTH, CV_8UC1, data);
			namedWindow("trainMat", WINDOW_NORMAL);
			setMouseCallback("trainMat", onMouseCallbackPointer);
			imshow("trainMat", trainMat);

			for (int i = 0; i < HEIGHT; i++)
				for (int j = 0; j < WIDTH; j++) {
					memblock[i * WIDTH + j] = data[i * WIDTH + j];
				}

			delete[] marks, data;

			//Testing data

			data = new uchar[HEIGHT * WIDTH * 1];
			fread(data, sizeof(uchar), HEIGHT * WIDTH * 1, testingFile);
			fclose(testingFile);

			Mat testMat = Mat(HEIGHT, WIDTH, CV_8UC1, data);
			namedWindow("testMat", WINDOW_NORMAL);
			imshow("testMat", testMat);

			for (int i = 0; i < HEIGHT; i++)
				for (int j = 0; j < WIDTH; j++) {
					testMemblock[i * WIDTH + j] = data[i * WIDTH + j];
				}

			delete[] data;

			marks = new uchar[HEIGHT * WIDTH * 1];
			fread(marks, sizeof(uchar), HEIGHT * WIDTH * 1, markingTestingFile);
			fclose(markingTestingFile);

			delete[] marks;

			Mat dst = Mat();

			inRange(trainMat, Scalar(75), Scalar(135), dst);

			namedWindow("dist", WINDOW_NORMAL);
			imshow("dist", dst);
		}

#pragma endregion
		private: System::Void button1_Click(System::Object^ sender, System::EventArgs^ e) {
			if (comboBox2->Text == "Test k-means") {
				testKmeans();
			}
			else if (comboBox2->Text == "Display histogramm") {
				displayHist();
			}
			else if (comboBox2->Text == "Extract features") {
				extractFeatures();
			}
			else if (comboBox2->Text == "Expansion of regions") {
				expansionRegions();
			}
			else if (comboBox2->Text == "Testing tool") {
				testTool();
			}
		}

	private: System::Void button3_Click(System::Object^ sender, System::EventArgs^ e) {
		OpenFileDialog openFileDialog;
		if (openFileDialog.ShowDialog() == System::Windows::Forms::DialogResult::OK)
		{
			displayPredictData(System::IO::Path::GetFileName(openFileDialog.FileName));
		}
	}
};
}