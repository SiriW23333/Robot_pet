#include "stm32f10x.h"                  // Device header
#include "Delay.h"
#include "OLED.h"
#include "BlueTooth.h"
#include "Servo.h"
#include "PetAction.h"
#include "Face_Config.h"
#include "PWM.h"


/***************************************************************************************
  * 本程序由博主我Sngels_wyh创建并免费开源共享
  * 你可以任意查看、使用和修改，并应用到自己的项目之中
  * 
  * 程序名称：				基于STM32F103C8T6单片机的桌面宠物小项目
  * 此程序更新时间：			2024.2.16
  * 
  * 主要在立创开源平台、CSDN网站、哔哩哔哩、抖音分享自己的作品。
	*	相关的资料可以去立创开源硬件平台网站上搜索,作品附件放置更新网盘，如有更新可方便快速查看
	* 地址:https://oshwhub.com/sngelswyh/stm32-smart-desktop-pet
  ***************************************************************************************/


uint16_t Time;
uint16_t HuXi;
uint16_t PanDuan=1;
uint16_t Wait=0;

int main(void)
{
	Servo_Init();
	OLED_Init();//OLED初始化
	BlueTooth_Init();//上位机端口初始化
	OLED_ShowImage(0,0,128,64,Face_sleep);
	OLED_Update();
	while(1)
	{	
		if(Action_Mode==0){Action_relaxed_getdowm();WServo_Angle(90);}//放松趴下
		else if(Action_Mode==1){Action_sit();}//坐下
		else if(Action_Mode==2){Action_upright();}//站立
		else if(Action_Mode==3){Action_getdowm();}//趴下
		else if(Action_Mode==4){Action_advance();}//前进
		else if(Action_Mode==5){Action_back();}//后退
		else if(Action_Mode==6){Action_Lrotation();}//左转
		else if(Action_Mode==7){Action_Rrotation();}//右转
		else if(Action_Mode==8){Action_Swing();}//摇摆
		else if(Action_Mode==9){Action_SwingTail();}//摇尾巴
		else if(Action_Mode==10){Action_JumpU();}//前跳
		else if(Action_Mode==11){Action_JumpD();}//后跳
		else if(Action_Mode==12){Action_upright2();}//站立方式2
		else if(Action_Mode==13){Action_Hello();}//打招呼
		else if(Action_Mode==14){Action_stretch();}//伸懒腰
		else if(Action_Mode==15){Action_Lstretch();}//后腿拉伸
		else if(Action_Mode==16){Action_stop();}//摇尾巴停止
		else if(Action_Mode==17){Action_advance();Action_Hello();Action_SwingTail();Action_Swing();Action_stop();Action_back();}//跳舞
	
	}
}


void TIM3_IRQHandler(void)
{
	if(TIM_GetITStatus(TIM3,TIM_IT_Update)==SET)
	{	
		if(AllLed==1 && BreatheLed==0)//如果灯光开启且不开启呼吸灯,亮度拉满
		{
			PWM_LED1(20000);
			PWM_LED2(20000);
		}
		else if(AllLed==1 && BreatheLed==1)//如果灯光开启且开启呼吸灯
		{
			if(PanDuan==1)
			{
				HuXi+=100;
				PWM_LED1(HuXi);
				PWM_LED2(HuXi);
				if(HuXi==20000)
					PanDuan=2;
			}
			else if(PanDuan==2)
			{
				HuXi-=100;
				PWM_LED1(HuXi);
				PWM_LED2(HuXi);
				if(HuXi==0)
				{
					PanDuan=3;
				}
			}
			else if(PanDuan==3)
			{
				Wait+=1000;
				if(Wait==20000)
				{
					PanDuan=1;
					Wait=0;
				}
			}
		}
		else if(AllLed==0)
		{
			PWM_LED1(0);
			PWM_LED2(0);
		}
		
		TIM_ClearITPendingBit(TIM3,TIM_IT_Update);
	}
}
