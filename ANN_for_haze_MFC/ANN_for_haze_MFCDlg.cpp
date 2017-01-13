// ANN_for_haze_MFCDlg.cpp : 实现文件
//

#include "stdafx.h"
#include "ANN_for_haze_MFC.h"
#include "ANN_for_haze_MFCDlg.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "cv.h"
#include "highgui.h"
#include <iostream>
#include<string> 
#include <cmath>
#include <iomanip>
#include "BP.h"

// Namespace for using cout.
using namespace std;

// Namespace for using Mat
using namespace cv;


#ifdef _DEBUG
#define new DEBUG_NEW
#endif

int max_value1, max_value2, min_value1, min_value2;
int max_label, min_label;
int original_of_AQI;
float rate1, edge_;

RGBQUAD VgaDefPal[256] = {                  //  VGA 256色默认调色板数据
	{0x00,0x00,0x00,0x00},{0xa8,0x00,0x00,0x00},{0x00,0xa8,0x00,0x00},{0xa8,0xa8,0x00,0x00},
	{0x00,0x00,0xa8,0x00},{0xa8,0x00,0xa8,0x00},{0x00,0x54,0xa8,0x00},{0xa8,0xa8,0xa8,0x00},
	{0x54,0x54,0x54,0x00},{0xfc,0x54,0x54,0x00},{0x54,0xfc,0x54,0x00},{0xfc,0xfc,0x54,0x00},
	{0x54,0x54,0xfc,0x00},{0xfc,0x54,0xfc,0x00},{0x54,0xfc,0xfc,0x00},{0xfc,0xfc,0xfc,0x00},
	{0x00,0x00,0x00,0x00},{0x14,0x14,0x14,0x00},{0x20,0x20,0x20,0x00},{0x2c,0x2c,0x2c,0x00},
	{0x38,0x38,0x38,0x00},{0x44,0x44,0x44,0x00},{0x50,0x50,0x50,0x00},{0x60,0x60,0x60,0x00},
	{0x70,0x70,0x70,0x00},{0x80,0x80,0x80,0x00},{0x90,0x90,0x90,0x00},{0xa0,0xa0,0xa0,0x00},
	{0xb4,0xb4,0xb4,0x00},{0xc8,0xc8,0xc8,0x00},{0xe0,0xe0,0xe0,0x00},{0xfc,0xfc,0xfc,0x00},
	{0xfc,0x00,0x00,0x00},{0xfc,0x00,0x40,0x00},{0xfc,0x00,0x7c,0x00},{0xfc,0x00,0xbc,0x00},
	{0xfc,0x00,0xfc,0x00},{0xbc,0x00,0xfc,0x00},{0x7c,0x00,0xfc,0x00},{0x40,0x00,0xfc,0x00},
	{0x00,0x00,0xfc,0x00},{0x00,0x40,0xfc,0x00},{0x00,0x7c,0xfc,0x00},{0x00,0xbc,0xfc,0x00},
	{0x00,0xfc,0xfc,0x00},{0x00,0xfc,0xbc,0x00},{0x00,0xfc,0x7c,0x00},{0x00,0xfc,0x40,0x00},
	{0x00,0xfc,0x00,0x00},{0x40,0xfc,0x00,0x00},{0x7c,0xfc,0x00,0x00},{0xbc,0xfc,0x00,0x00},
	{0xfc,0xfc,0x00,0x00},{0xfc,0xbc,0x00,0x00},{0xfc,0x7c,0x00,0x00},{0xfc,0x40,0x00,0x00},
	{0xfc,0x7c,0x7c,0x00},{0xfc,0x7c,0x9c,0x00},{0xfc,0x7c,0xbc,0x00},{0xfc,0x7c,0xdc,0x00},
	{0xfc,0x7c,0xfc,0x00},{0xdc,0x7c,0xfc,0x00},{0xbc,0x7c,0xfc,0x00},{0x9c,0x7c,0xfc,0x00},
	{0x7c,0x7c,0xfc,0x00},{0x7c,0x9c,0xfc,0x00},{0x7c,0xbc,0xfc,0x00},{0x7c,0xdc,0xfc,0x00},
	{0x7c,0xfc,0xfc,0x00},{0x7c,0xfc,0xdc,0x00},{0x7c,0xfc,0xbc,0x00},{0x7c,0xfc,0x9c,0x00},
	{0x7c,0xfc,0x7c,0x00},{0x9c,0xfc,0x7c,0x00},{0xbc,0xfc,0x7c,0x00},{0xdc,0xfc,0x7c,0x00},
	{0xfc,0xfc,0x7c,0x00},{0xfc,0xdc,0x7c,0x00},{0xfc,0xbc,0x7c,0x00},{0xfc,0x9c,0x7c,0x00},
	{0xfc,0xb4,0xb4,0x00},{0xfc,0xb4,0xc4,0x00},{0xfc,0xb4,0xd8,0x00},{0xfc,0xb4,0xe8,0x00},
	{0xfc,0xb4,0xfc,0x00},{0xe8,0xb4,0xfc,0x00},{0xd8,0xb4,0xfc,0x00},{0xc4,0xb4,0xfc,0x00},
	{0xb4,0xb4,0xfc,0x00},{0xb4,0xc4,0xfc,0x00},{0xb4,0xd8,0xfc,0x00},{0xb4,0xe8,0xfc,0x00},
	{0xb4,0xfc,0xfc,0x00},{0xb4,0xfc,0xe8,0x00},{0xb4,0xfc,0xd8,0x00},{0xb4,0xfc,0xc4,0x00},
	{0xb4,0xfc,0xb4,0x00},{0xc4,0xfc,0xb4,0x00},{0xd8,0xfc,0xb4,0x00},{0xe8,0xfc,0xb4,0x00},
	{0xfc,0xfc,0xb4,0x00},{0xfc,0xe8,0xb4,0x00},{0xfc,0xd8,0xb4,0x00},{0xfc,0xc4,0xb4,0x00},
	{0x70,0x00,0x00,0x00},{0x70,0x00,0x1c,0x00},{0x70,0x00,0x38,0x00},{0x70,0x00,0x54,0x00},
	{0x70,0x00,0x70,0x00},{0x54,0x00,0x70,0x00},{0x38,0x00,0x70,0x00},{0x1c,0x00,0x70,0x00},
	{0x00,0x00,0x70,0x00},{0x00,0x1c,0x70,0x00},{0x00,0x38,0x70,0x00},{0x00,0x54,0x70,0x00},
	{0x00,0x70,0x70,0x00},{0x00,0x70,0x54,0x00},{0x00,0x70,0x38,0x00},{0x00,0x70,0x1c,0x00},
	{0x00,0x70,0x00,0x00},{0x1c,0x70,0x00,0x00},{0x38,0x70,0x00,0x00},{0x54,0x70,0x00,0x00},
	{0x70,0x70,0x00,0x00},{0x70,0x54,0x00,0x00},{0x70,0x38,0x00,0x00},{0x70,0x1c,0x00,0x00},
	{0x70,0x38,0x38,0x00},{0x70,0x38,0x44,0x00},{0x70,0x38,0x54,0x00},{0x70,0x38,0x60,0x00},
	{0x70,0x38,0x70,0x00},{0x60,0x38,0x70,0x00},{0x54,0x38,0x70,0x00},{0x44,0x38,0x70,0x00},
	{0x38,0x38,0x70,0x00},{0x38,0x44,0x70,0x00},{0x38,0x54,0x70,0x00},{0x38,0x60,0x70,0x00},
	{0x38,0x70,0x70,0x00},{0x38,0x70,0x60,0x00},{0x38,0x70,0x54,0x00},{0x38,0x70,0x44,0x00},
	{0x38,0x70,0x38,0x00},{0x44,0x70,0x38,0x00},{0x54,0x70,0x38,0x00},{0x60,0x70,0x38,0x00},
	{0x70,0x70,0x38,0x00},{0x70,0x60,0x38,0x00},{0x70,0x54,0x38,0x00},{0x70,0x44,0x38,0x00},
	{0x70,0x50,0x50,0x00},{0x70,0x50,0x58,0x00},{0x70,0x50,0x60,0x00},{0x70,0x50,0x68,0x00},
	{0x70,0x50,0x70,0x00},{0x68,0x50,0x70,0x00},{0x60,0x50,0x70,0x00},{0x58,0x50,0x70,0x00},
	{0x50,0x50,0x70,0x00},{0x50,0x58,0x70,0x00},{0x50,0x60,0x70,0x00},{0x50,0x68,0x70,0x00},
	{0x50,0x70,0x70,0x00},{0x50,0x70,0x68,0x00},{0x50,0x70,0x60,0x00},{0x50,0x70,0x58,0x00},
	{0x50,0x70,0x50,0x00},{0x58,0x70,0x50,0x00},{0x60,0x70,0x50,0x00},{0x68,0x70,0x50,0x00},
	{0x70,0x70,0x50,0x00},{0x70,0x68,0x50,0x00},{0x70,0x60,0x50,0x00},{0x70,0x58,0x50,0x00},
	{0x40,0x00,0x00,0x00},{0x40,0x00,0x10,0x00},{0x40,0x00,0x20,0x00},{0x40,0x00,0x30,0x00},
	{0x40,0x00,0x40,0x00},{0x30,0x00,0x40,0x00},{0x20,0x00,0x40,0x00},{0x10,0x00,0x40,0x00},
	{0x00,0x00,0x40,0x00},{0x00,0x10,0x40,0x00},{0x00,0x20,0x40,0x00},{0x00,0x30,0x40,0x00},
	{0x00,0x40,0x40,0x00},{0x00,0x40,0x30,0x00},{0x00,0x40,0x20,0x00},{0x00,0x40,0x10,0x00},
	{0x00,0x40,0x00,0x00},{0x10,0x40,0x00,0x00},{0x20,0x40,0x00,0x00},{0x30,0x40,0x00,0x00},
	{0x40,0x40,0x00,0x00},{0x40,0x30,0x00,0x00},{0x40,0x20,0x00,0x00},{0x40,0x10,0x00,0x00},
	{0x40,0x20,0x20,0x00},{0x40,0x20,0x28,0x00},{0x40,0x20,0x30,0x00},{0x40,0x20,0x38,0x00},
	{0x40,0x20,0x40,0x00},{0x38,0x20,0x40,0x00},{0x30,0x20,0x40,0x00},{0x28,0x20,0x40,0x00},
	{0x20,0x20,0x40,0x00},{0x20,0x28,0x40,0x00},{0x20,0x30,0x40,0x00},{0x20,0x38,0x40,0x00},
	{0x20,0x40,0x40,0x00},{0x20,0x40,0x38,0x00},{0x20,0x40,0x30,0x00},{0x20,0x40,0x28,0x00},
	{0x20,0x40,0x20,0x00},{0x28,0x40,0x20,0x00},{0x30,0x40,0x20,0x00},{0x38,0x40,0x20,0x00},
	{0x40,0x40,0x20,0x00},{0x40,0x38,0x20,0x00},{0x40,0x30,0x20,0x00},{0x40,0x28,0x20,0x00},
	{0x40,0x2c,0x2c,0x00},{0x40,0x2c,0x30,0x00},{0x40,0x2c,0x34,0x00},{0x40,0x2c,0x3c,0x00},
	{0x40,0x2c,0x40,0x00},{0x3c,0x2c,0x40,0x00},{0x34,0x2c,0x40,0x00},{0x30,0x2c,0x40,0x00},
	{0x2c,0x2c,0x40,0x00},{0x2c,0x30,0x40,0x00},{0x2c,0x34,0x40,0x00},{0x2c,0x3c,0x40,0x00},
	{0x2c,0x40,0x40,0x00},{0x2c,0x40,0x3c,0x00},{0x2c,0x40,0x34,0x00},{0x2c,0x40,0x30,0x00},
	{0x2c,0x40,0x2c,0x00},{0x30,0x40,0x2c,0x00},{0x34,0x40,0x2c,0x00},{0x3c,0x40,0x2c,0x00},
	{0x40,0x40,0x2c,0x00},{0x40,0x3c,0x2c,0x00},{0x40,0x34,0x2c,0x00},{0x40,0x30,0x2c,0x00},
	{0xa8,0xa8,0xa8,0x00},{0x54,0xfc,0xfc,0x00},{0xfc,0x54,0xfc,0x00},{0xfc,0xfc,0x54,0x00},
	{0xfc,0x54,0x54,0x00},{0x54,0xfc,0x54,0x00},{0x54,0x54,0xfc,0x00},{0xfc,0xfc,0xfc,0x00}  
};


// 用于应用程序“关于”菜单项的 CAboutDlg 对话框

class CAboutDlg : public CDialog
{
public:
	CAboutDlg();

// 对话框数据
	enum { IDD = IDD_ABOUTBOX };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

// 实现
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialog(CAboutDlg::IDD)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialog)
END_MESSAGE_MAP()


// CANN_for_haze_MFCDlg 对话框




CANN_for_haze_MFCDlg::CANN_for_haze_MFCDlg(CWnd* pParent /*=NULL*/)
	: CDialog(CANN_for_haze_MFCDlg::IDD, pParent)
	, m_Edit_times(0)
	, m_Edit_number(0)
	, m_Edit_original_AQI(0)
	, m_Edit_original_class(0)
	, m_Edit_predict_AQI(0)
	, m_Edit_predict_class(_T(""))
	, m_Edit_original_class2(_T(""))
	, m_Edit_error_AQI(0)
	, m_Edit_error_train(0)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CANN_for_haze_MFCDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	DDX_Text(pDX, IDC_EDIT1_times, m_Edit_times);
	DDX_Text(pDX, IDC_EDIT2_number_test, m_Edit_number);
	DDX_Text(pDX, IDC_EDIT3_original_AQI, m_Edit_original_AQI);
	//DDX_Text(pDX, IDC_EDIT4_original_class, m_Edit_original_class);
	DDX_Text(pDX, IDC_EDIT5_predict_AQI, m_Edit_predict_AQI);
	DDX_Text(pDX, IDC_EDIT6_predict_class, m_Edit_predict_class);
	DDX_Text(pDX, IDC_EDIT8_original_class, m_Edit_original_class2);
	DDX_Text(pDX, IDC_EDIT7_error_AQI, m_Edit_error_AQI);
	DDX_Control(pDX, IDC_STATIC_test_picture, m_Edit_picture);
	DDX_Text(pDX, IDC_EDIT1_error_train, m_Edit_error_train);
}

BEGIN_MESSAGE_MAP(CANN_for_haze_MFCDlg, CDialog)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	//}}AFX_MSG_MAP
	ON_BN_CLICKED(IDC_BUTTON_train, &CANN_for_haze_MFCDlg::OnBnClickedButtontrain)
	ON_BN_CLICKED(IDC_BUTTON3_input_number, &CANN_for_haze_MFCDlg::OnBnClickedButton3inputnumber)
	ON_BN_CLICKED(IDC_BUTTON_test, &CANN_for_haze_MFCDlg::OnBnClickedButtontest)
END_MESSAGE_MAP()


// CANN_for_haze_MFCDlg 消息处理程序

BOOL CANN_for_haze_MFCDlg::OnInitDialog()
{
	CDialog::OnInitDialog();

	// 将“关于...”菜单项添加到系统菜单中。

	// IDM_ABOUTBOX 必须在系统命令范围内。
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		CString strAboutMenu;
		strAboutMenu.LoadString(IDS_ABOUTBOX);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 设置此对话框的图标。当应用程序主窗口不是对话框时，框架将自动
	//  执行此操作
	SetIcon(m_hIcon, TRUE);			// 设置大图标
	SetIcon(m_hIcon, FALSE);		// 设置小图标

	// TODO: 在此添加额外的初始化代码

	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
}

void CANN_for_haze_MFCDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialog::OnSysCommand(nID, lParam);
	}
}

// 如果向对话框添加最小化按钮，则需要下面的代码
//  来绘制该图标。对于使用文档/视图模型的 MFC 应用程序，
//  这将由框架自动完成。

void CANN_for_haze_MFCDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 用于绘制的设备上下文

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 使图标在工作区矩形中居中
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 绘制图标
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialog::OnPaint();
	}
}

//当用户拖动最小化窗口时系统调用此函数取得光标
//显示。
HCURSOR CANN_for_haze_MFCDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



// 用于将IplImage显示到对话框上面，建立位图信息
LPBITMAPINFO CANN_for_haze_MFCDlg::CtreateMapInfo(IplImage* workImg , int flag)
{
	BITMAPINFOHEADER BIH={40,1,1,1,8,0,0,0,0,0,0};
	LPBITMAPINFO lpBmi;
	int      wid,hei,bits,colors,i;
	RGBQUAD  ColorTab[256];

	wid =workImg->width;
	hei =workImg->height;
	bits=workImg->depth*workImg->nChannels;

	if (bits>8) colors=0;
	else colors=1<<bits;

	lpBmi = (LPBITMAPINFO) malloc(40+4*colors);
	BIH.biWidth    =  wid;
	BIH.biHeight   = - hei;          //位图和IPLImage* 的图像有反转的关系，up-bottom就是负的，bottom-up就是正的
	BIH.biBitCount = (BYTE) bits;
	memcpy(lpBmi, &BIH, 40);                   //  复制位图信息头

	if (bits==8) 
	{                                         //  256 色位图
		if (flag==1) 
		{                                       //  设置灰阶调色板
			for (i=0;i<256;i++) 
			{
				ColorTab[i].rgbRed=ColorTab[i].rgbGreen=
					ColorTab[i].rgbBlue=(BYTE) i;
			}
			memcpy(lpBmi->bmiColors,ColorTab,1024);
		}
		else if (flag==2) 
		{                                        //  设置默认调色板
			memcpy(lpBmi->bmiColors,VgaDefPal,1024);
		}
	}
	return(lpBmi);
}

void CANN_for_haze_MFCDlg::DrawImage1(IplImage* Image)
{
	CWnd* pWnd = GetDlgItem(IDC_STATIC_test_picture);
	CDC* pDC = pWnd->GetDC(); 
	HDC hDC2 = pDC->GetSafeHdc();

	CRect rect;
	m_Edit_picture.GetWindowRect(&rect);

	//CRect rect1;
	//m_pDis2Wnd->GetWindowRect(&rect1);

	CDC m_dcMem;
	HDC hMDC;
	CBitmap m_bmp;
	m_dcMem.CreateCompatibleDC(pDC);
	hMDC = m_dcMem.GetSafeHdc();
	m_bmp.CreateCompatibleBitmap(pDC, rect.Width(), rect.Height());
	m_dcMem.SelectObject(&m_bmp); 

	int m_dibFlag;
	LPBITMAPINFO m_lpBmi;
	char *pBits;
	m_dibFlag = 1;
	m_lpBmi = CtreateMapInfo( Image, m_dibFlag);
	m_dibFlag=0;

	if (Image)  pBits = Image->imageData;
	//else AfxMessageBox("image error");
	//cvFlip(Image,NULL,2);
	//cvFlip(Image, NULL, 0);

	if (Image) {

		SetStretchBltMode(hMDC,COLORONCOLOR);//  刷新窗口画面
		StretchDIBits(hMDC,
			0,0,rect.Width(),rect.Height(),
			0,0,Image->width,Image->height,
			pBits,m_lpBmi,DIB_RGB_COLORS,SRCCOPY);
	}
	pDC->BitBlt(0,0,rect.Width(), rect.Height(),
		&m_dcMem,0, 0, SRCCOPY);
	free(m_lpBmi);
	HDC DeleteDC(hMDC);
	m_dcMem.DeleteDC();
	m_bmp.DeleteObject();
	ReleaseDC(pDC);
}


//定义训练阶段函数train_step()
void CANN_for_haze_MFCDlg::train_step()
{
	string read_image_path;
	int frame_num = 0;
	read_image_path = "C:\\Users\\LDY\\Desktop\\samples\\train_samples\\";
	string image_name;
	char c[10];
	float result1[34];
	float result2[34];
	FILE *stream1;
	FILE *stream2;

	for (int pic=0; pic<34; pic++)
	{
		//1. 读入sample文件夹内的图像
		frame_num = frame_num + 1;
		sprintf(c,"%d.jpg",frame_num);
		image_name = c;
		image_name = read_image_path + image_name;
		//cout<<image_name<<endl;

		const char *t = image_name.c_str();
		IplImage* image=cvLoadImage(t,-1);
		cvResize(image,image,CV_INTER_LINEAR);  
		//cvNamedWindow("...",CV_WINDOW_AUTOSIZE);
		//cvShowImage("...",image);
		IplImage* image1 = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
		IplImage *pCannyImg=cvCreateImage(cvGetSize(image1),IPL_DEPTH_8U,1);


		//2. 提取第一维特征： white rate of histgram
		cvCvtColor(image,image1,CV_BGR2GRAY);
		int sum_pixel = 0;
		int sum_white_pixel = 0;
		double rate = 0;
		CvScalar s;
		for (int i=0;i<image1->height;i++)
		{
			for (int j=0;j<image1->width;j++)
			{
				sum_pixel = sum_pixel + 1;
				s = cvGet2D(image1,i,j);
				if (s.val[0]>127)
				{
					sum_white_pixel = sum_white_pixel + 1;
				}
			}
		}		
		rate=sum_white_pixel*100/sum_pixel;
		result1[pic] = rate;      //存放所有图像的第一维特征
		//cout<<"sum: "<<sum_pixel<<endl<<"sum_white: "<<sum_white_pixel<<endl;
		//cout<<"rate: "<<rate<<endl;


		//3. 提取第二维特征: edge
		cvCanny(image1,pCannyImg,50,150,3);
		int sum_pixel_edge = 0;
		CvScalar s1;
		for (int i=0;i<pCannyImg->height;i++)
		{
			for (int j=0;j<pCannyImg->width;j++)
			{
				s1 = cvGet2D(pCannyImg,i,j);
				if (s1.val[0] != 0)
				{
					sum_pixel_edge = sum_pixel_edge + 1;
				}
			}
		}
		result2[pic] = sum_pixel_edge;  //存放所有图像的第二维特征
		//cout<<"edge: "<<sum_pixel_edge<<endl<<endl;
		cvReleaseImage(&image);
		cvReleaseImage(&image1);
		cvReleaseImage(&pCannyImg);
	}

	//4. 训练特征的归一化   （  所有归一化使用的公式： X* = (x-min)/(max-min)  ）
	//int max_value1, max_value2, min_value1, min_value2;
	int max_1 =0;
	int max_2 =0;
	int min_1 =0; 
	int min_2 =0;
	for (int i=1; i<34; i++)
	{
		if (result1[i] > result1[max_1])
		{
			max_1 = i;
		}
		if (result2[i] > result2[max_2])
		{
			max_2 = i;
		}
	}
	max_value1 = result1[max_1];
	max_value2 = result2[max_2];

	for (int i=1; i<34; i++)
	{
		if (result1[i] < result1[min_1])
		{
			min_1 = i;
		}
		if (result2[i] < result2[min_2])
		{
			min_2 = i;
		}
	}
	min_value1 = result1[min_1];
	min_value2 = result2[min_2];

	for (int i=0; i<34; i++)
	{
		result1[i] = (result1[i] - min_value1)/(max_value1 - min_value1);   //存放第一维特征归一化结果
		//cout<<".........."<<result1[i]<<endl;
		result2[i] = (result2[i] - min_value2)/(max_value2 - min_value2);   //存放第二维特征归一化结果
		//cout<<"..........fsfsfsf::"<<result2[i]<<endl;
	}


	//5. 将归一化后的训练特征数据保存在“.txt”中
	if(( stream1 = fopen("C:\\Users\\LDY\\Desktop\\samples\\train_samples\\eigenvalue_white_rate.txt", "w+" ))==NULL)   //存放第一维特征的.txt
	{
		cout<<"创建文件失败!";
		exit(1);
	}
	for (int i=0; i<34; i++)
	{
		fprintf(stream1, "%f\n", result1[i]);
	}
	fclose(stream1);

	if(( stream2 = fopen("C:\\Users\\LDY\\Desktop\\samples\\train_samples\\eigenvalue_edge.txt", "w+" ))==NULL)   //存放第二维特征的.txt
	{
		cout<<"创建文件失败!";
		exit(1);
	}
	for (int i=0; i<34; i++)
	{
		fprintf(stream2, "%f\n", result2[i]);
	}
	fclose(stream2);


	//6. 归一化样本标签(即特征所对应的输出值AQI)
	float eigenvalue1[34][1];
	float label[34];  //样本标签归一化结果放在label[ ]里面
	FILE *stream3;
	FILE *stream4;
	if(( stream3 = fopen("C:\\Users\\LDY\\Desktop\\samples\\train_samples\\train_samples_out.txt", "r" ))==NULL)
	{
		cout<<"打开文件失败!";
		exit(1);
	}
	for (int k=0; k<34; k++)
	{
		fscanf(stream3, "%f", &eigenvalue1[k][0]);
		label[k] = eigenvalue1[k][0];
	}
	fclose(stream3);

	int lable_max = 0;
	int label_min = 0;
	//int max_label, min_label;

	for (int i=1; i<34; i++)
	{
		if (label[i] > label[lable_max])
		{
			lable_max = i;
		}
	}
	max_label = label[lable_max];

	for (int i=1; i<34; i++)
	{
		if (label[i] < label[label_min])
		{
			label_min = i;
		}
	}
	min_label = label[label_min];

	//7. 将归一化后的样本标签数据保存在“.txt”中
	for (int i=0; i<34; i++)
	{
		label[i] = (label[i] - min_label)/(max_label - min_label);
	}

	if(( stream4 = fopen("C:\\Users\\LDY\\Desktop\\samples\\train_samples\\train_samples_out_normalization.txt", "w+" ))==NULL)
	{
		cout<<"创建文件失败!";
		exit(1);
	}
	for (int i=0; i<34; i++)
	{
		fprintf(stream4, "%f\n", label[i]);
	}
	fclose(stream4);


	//8.用预处理好的训练数据及特征来训练BP神经网络。
	float eigenvalue1_[34][1];
	float eigenvalue2_[34][1];
	float out[34][1];
	double X[trainsample][innode];
	double Y[trainsample][outnode];
	FILE *stream_in1;
	FILE *stream_in2;
	FILE *stream_out;
	int k,l;

	if(( stream_in1 = fopen("C:\\Users\\LDY\\Desktop\\samples\\train_samples\\eigenvalue_white_rate.txt", "r" ))==NULL)
	{
		cout<<"打开文件失败!";
		exit(1);
	}
	if(( stream_in2 = fopen("C:\\Users\\LDY\\Desktop\\samples\\train_samples\\eigenvalue_edge.txt", "r" ))==NULL)
	{
		cout<<"打开文件失败!";
		exit(1);
	}
	if(( stream_out = fopen("C:\\Users\\LDY\\Desktop\\samples\\train_samples\\train_samples_out_normalization.txt", "r" ))==NULL)
	{
		cout<<"打开文件失败!";
		exit(1);
	}
	for (k=0; k<34; k++)
	{
		fscanf(stream_in1, "%f", &eigenvalue1_[k][0]);
		fscanf(stream_in2, "%f", &eigenvalue2_[k][0]);
		fscanf(stream_out, "%f", &out[k][0]);
		//cout<<"result0: "<<result0[k][0]<<endl<<"result1: "<<result1[k][0]<<endl<<endl;

		X[k][0] = eigenvalue1_[k][0];
		X[k][1] = eigenvalue2_[k][0];
		Y[k][0] = out[k][0];
		//cout<<"x: "<<X[k][0]<<endl<<"y: "<<Y[k][0]<<endl<<endl;
	}
	fclose(stream_in1);
	fclose(stream_in2);
	fclose(stream_out);

	BpNet bp;
	bp.init();
	int times=0;
	while(bp.error > 0.06)    //用于设定迭代过程中的误差！！！！！！！！！！！！！！！！！！！！！！！
	{
		bp.e=0.0;
		times++;
		bp.train(X,Y);   //将输入的第一维特征X和第二维特征Y放入BP网络进行训练
		m_Edit_times = times;
		m_Edit_error_train = bp.error;
		UpdateData(FALSE);
		cout<<"Times="<<times<<" error="<<bp.error<<endl;
	}
	
	bp.writetrain();   //将训练出的模型写入.txt
	cout<<"训练完成！"<<endl<<endl<<endl<<"进入测试阶段 "<<endl;

}


//定义测试阶段函数test_step() and test_step2()
void CANN_for_haze_MFCDlg::test_step()
{
	//对测试图像提取特征 and 归一化 
	string read_image_path1;
	int frame_num1 = 0;   //测试样本的图像编号输入在这*****************************************************************
	read_image_path1 = "C:\\Users\\LDY\\Desktop\\samples\\test_samples\\";
	string image_name1;
	char c1[10];
	cout<<"请输入想要测试的图片序号： ";
	UpdateData(TRUE);
	if (m_Edit_number == 0)
	{
		frame_num1 = 1;
	}
	frame_num1 = m_Edit_number;


	float out_test[9][1];
	float label_test[39];  //样本标签归一化结果放在label[ ]里面
	FILE *stream_out_test;
	if(( stream_out_test = fopen("C:\\Users\\LDY\\Desktop\\samples\\test_samples\\test_samples_out.txt", "r" ))==NULL)
	{
		cout<<"打开文件失败!";
		exit(1);
	}
	for (int k=0; k<34; k++)
	{
		fscanf(stream_out_test, "%f", &out_test[k][0]);
		label_test[k] = out_test[k][0];
	}
	fclose(stream_out_test);

	CString merit("优");
	CString fine("良");
	CString soft("轻度污染");
	CString medium("中度污染");
	CString heavy("重度污染");
	CString critical("严重污染");
	int frame_num1_test;
	//int original_of_AQI;
	frame_num1_test = frame_num1 - 1;
	original_of_AQI = label_test[frame_num1_test];
	cout<<"测试图像的原始AQI: "<<original_of_AQI<<'\t'<<'\t';   //输出测试样本的原始AQI
	m_Edit_original_AQI = original_of_AQI;   //输出测试样本的原始AQI
	UpdateData(FALSE);

	if (original_of_AQI <= 50)
	{
		cout<<"测试图像的原始空气质量级别为：优"<<endl;
		m_Edit_original_class2 = merit;
		UpdateData(FALSE);
	}
	else if (original_of_AQI <= 100)
	{
		cout<<"测试图像的原始空气质量级别为：良"<<endl;
		m_Edit_original_class2 = fine;
		UpdateData(FALSE);
	} 
	else if (original_of_AQI <= 150)
	{
		cout<<"测试图像的原始空气质量级别为：轻度污染"<<endl;
		m_Edit_original_class2 = soft;
		UpdateData(FALSE);
	} 
	else if (original_of_AQI <= 200)
	{
		cout<<"测试图像的原始空气质量级别为：中度污染"<<endl;
		m_Edit_original_class2 = medium;
		UpdateData(FALSE);
	}
	else if (original_of_AQI <= 300)
	{
		cout<<"测试图像的原始空气质量级别为：重度污染"<<endl;
		m_Edit_original_class2 = heavy;
		UpdateData(FALSE);
	}
	else
	{
		cout<<"测试图像的原始空气质量级别为：严重污染"<<endl;
		m_Edit_original_class2 = critical;
		UpdateData(FALSE);
	}

	sprintf(c1,"%d.jpg",frame_num1);
	image_name1 = c1;
	image_name1 = read_image_path1 + image_name1;
	const char *t1 = image_name1.c_str();
	IplImage* image_=cvLoadImage(t1,-1);
	cvResize(image_,image_,CV_INTER_LINEAR);  

	DrawImage1(image_);  //显示测试图像！！！！！！！！！！！！！

	//cvNamedWindow("测试图像",CV_WINDOW_AUTOSIZE);
	//cvShowImage("测试图像",image_);
	IplImage* image1_ = cvCreateImage(cvGetSize(image_), IPL_DEPTH_8U, 1);
	IplImage *pCannyImg_=cvCreateImage(cvGetSize(image1_),IPL_DEPTH_8U,1);

	//first method of eigenvalue extraction: white rate of histgram
	cvCvtColor(image_,image1_,CV_BGR2GRAY);
	int sum_pixel1 = 0;
	int sum_white_pixel1 = 0;
	CvScalar s_;
	for (int i=0;i<image1_->height;i++)
	{
		for (int j=0;j<image1_->width;j++)
		{
			sum_pixel1 = sum_pixel1 + 1;
			s_ = cvGet2D(image1_,i,j);
			if (s_.val[0]>127)
			{
				sum_white_pixel1 = sum_white_pixel1 + 1;
			}
		}
	}		
	rate1 = sum_white_pixel1*100/sum_pixel1;

	//second method of eigenvalue extraction: edge
	cvCanny(image1_,pCannyImg_,50,150,3);
	int sum_pixel_edge1 = 0;
	CvScalar s1_;
	for (int i=0;i<pCannyImg_->height;i++)
	{
		for (int j=0;j<pCannyImg_->width;j++)
		{
			s1_ = cvGet2D(pCannyImg_,i,j);
			if (s1_.val[0] != 0)
			{
				sum_pixel_edge1 = sum_pixel_edge1 + 1;
			}
		}
	}

	//测试特征归一化
	rate1 = (rate1 - min_value1)/(max_value1 - min_value1);
	edge_ = (float)(sum_pixel_edge1 - min_value2)/(max_value2 - min_value2);
	//cout<<"rate_test_normalization: "<<rate1<<endl;
	//cout<<"edge_test_normalization: "<<edge_<<endl<<endl;
	cvReleaseImage(&image_);
	cvReleaseImage(&image1_);
	cvReleaseImage(&pCannyImg_);
}


void CANN_for_haze_MFCDlg::test_step2()
{
	//开始进行测试阶段
	BpNet bp;
	bp.readtrain();    //从.txt读出已经训练的模型
	double m[innode]={rate1,edge_};  //输入测试图像的归一化后的特征***********************************************************************
	double *r=bp.recognize(m);
	double bp_result;
	for(int i=0;i<outnode;++i)
	{
		bp_result = bp.result[i];
		//cout<<"the result of bp_ANN: "<<bp_result<<endl; //输出测试样本的预测AQI
	}		

	CString merit("优");
	CString fine("良");
	CString soft("轻度污染");
	CString medium("中度污染");
	CString heavy("重度污染");
	CString critical("严重污染");
	float result_of_AQI;
	result_of_AQI = (max_label-min_label)*bp.result[0] - min_label;  //通过反归一化公式计算出相应的预测AQI值
	cout<<"测试图像的预测AQI: "<<result_of_AQI<<'\t'; //输出测试样本的预测AQI
	m_Edit_predict_AQI = result_of_AQI;   
	UpdateData(FALSE);
	if (result_of_AQI <= 50)
	{
		cout<<"测试图像的原始空气质量级别为：优"<<endl;
		m_Edit_predict_class = merit;
		UpdateData(FALSE);
	}
	else if (result_of_AQI <= 100)
	{
		cout<<"测试图像的原始空气质量级别为：良"<<endl;
		m_Edit_predict_class = fine;
		UpdateData(FALSE);
	} 
	else if (result_of_AQI <= 150)
	{
		cout<<"测试图像的原始空气质量级别为：轻度污染"<<endl;
		m_Edit_predict_class = soft;
		UpdateData(FALSE);
	} 
	else if (result_of_AQI <= 200)
	{
		cout<<"测试图像的原始空气质量级别为：中度污染"<<endl;
		m_Edit_predict_class = medium;
		UpdateData(FALSE);
	}
	else if (result_of_AQI <= 300)
	{
		cout<<"测试图像的原始空气质量级别为：重度污染"<<endl;
		m_Edit_predict_class = heavy;
		UpdateData(FALSE);
	}
	else
	{
		cout<<"测试图像的原始空气质量级别为：严重污染"<<endl;
		m_Edit_predict_class = critical;
		UpdateData(FALSE);
	}
	//计算并显示误差
	float error_AQI;
	error_AQI = abs(original_of_AQI - result_of_AQI)*100/original_of_AQI;
	cout<<"AQI误差为："<<error_AQI<<"%"<<endl;
	m_Edit_error_AQI = error_AQI;
	UpdateData(FALSE);

}



void CANN_for_haze_MFCDlg::OnBnClickedButtontrain()
{
	// TODO: 在此添加控件通知处理程序代码
	train_step();
}

void CANN_for_haze_MFCDlg::OnBnClickedButton3inputnumber()
{
	// TODO: 在此添加控件通知处理程序代码
	test_step();
}

void CANN_for_haze_MFCDlg::OnBnClickedButtontest()
{
	// TODO: 在此添加控件通知处理程序代码
	test_step2();
}
