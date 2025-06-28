#include "stm32f10x.h"                  // Device header
#include "PWM.h"
#include "PetAction.h"
#include "Face_Config.h"

uint16_t AllLed=1;  //开启灯光
uint16_t BreatheLed=0;//开启呼吸灯
uint16_t Sustainedmove=0;//持续运动


uint16_t Action_Mode=0;
uint16_t SpeedDelay=200;
uint16_t SwingDelay=3;
uint16_t Face_Mode=0;
uint8_t WeiBa=0;

void BlueTooth_Init(void)
{
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA,ENABLE);//开启GPIOA时钟
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_USART1,ENABLE);//开启串口时钟
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOB,ENABLE);//开启GPIOB时钟
	RCC_APB1PeriphClockCmd(RCC_APB1Periph_USART3,ENABLE);//开启串口时钟
	//语音
	GPIO_InitTypeDef GPIO_InitStructure;
	GPIO_InitStructure.GPIO_Mode=GPIO_Mode_AF_PP;//复用推挽输出模式
	GPIO_InitStructure.GPIO_Pin=GPIO_Pin_9;//默认PA9是USART1_TX的复用
	GPIO_InitStructure.GPIO_Speed=GPIO_Speed_50MHz;
	GPIO_Init(GPIOA,&GPIO_InitStructure);
	GPIO_InitStructure.GPIO_Mode=GPIO_Mode_IPU;//复用上拉输入模式
	GPIO_InitStructure.GPIO_Pin=GPIO_Pin_10;//默认PA9是USART1_RX的复用
	GPIO_InitStructure.GPIO_Speed=GPIO_Speed_50MHz;
	GPIO_Init(GPIOA,&GPIO_InitStructure);
	//蓝牙
	GPIO_InitStructure.GPIO_Mode=GPIO_Mode_AF_PP;//复用推挽输出模式
	GPIO_InitStructure.GPIO_Pin=GPIO_Pin_10;//默认PB10是USART3_TX的复用
	GPIO_InitStructure.GPIO_Speed=GPIO_Speed_50MHz;
	GPIO_Init(GPIOB,&GPIO_InitStructure);
	GPIO_InitStructure.GPIO_Mode=GPIO_Mode_IPU;//复用上拉输入模式
	GPIO_InitStructure.GPIO_Pin=GPIO_Pin_11;//默认PB11是USART3_RX的复用
	GPIO_InitStructure.GPIO_Speed=GPIO_Speed_50MHz;
	GPIO_Init(GPIOB,&GPIO_InitStructure);
	
	USART_InitTypeDef UASRT_InitStructure;//USART初始化
	UASRT_InitStructure.USART_BaudRate=9600;//波特率9600
	UASRT_InitStructure.USART_HardwareFlowControl=USART_HardwareFlowControl_None;//不需要硬件流控制
	UASRT_InitStructure.USART_Mode=USART_Mode_Rx|USART_Mode_Tx;//接受与发送均打开
	UASRT_InitStructure.USART_Parity=USART_Parity_No;//不需要奇偶校验
	UASRT_InitStructure.USART_StopBits=USART_StopBits_1;//停止位为1
	UASRT_InitStructure.USART_WordLength=USART_WordLength_8b;//字长8位
	USART_Init(USART1,&UASRT_InitStructure);
	USART_Init(USART3,&UASRT_InitStructure);
	
	USART_ITConfig(USART1,USART_IT_RXNE,ENABLE);//语音接收中断配置，也就是如果接送到消息就直接中断
	USART_ITConfig(USART3,USART_IT_RXNE,ENABLE);//蓝牙接收中断配置，也就是如果接送到消息就直接中断
	
	//中断
	NVIC_PriorityGroupConfig(NVIC_PriorityGroup_2);//分组2
	//语音中断配置
	NVIC_InitTypeDef NVIC_InitStructure;
	NVIC_InitStructure.NVIC_IRQChannel=USART1_IRQn;//特定的通道
	NVIC_InitStructure.NVIC_IRQChannelCmd=ENABLE;//通道使能
	NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority=1;//抢占式优先级为1
	NVIC_InitStructure.NVIC_IRQChannelSubPriority=1;//响应优先级为1
	NVIC_Init(&NVIC_InitStructure);
	//蓝牙中断配置
	NVIC_InitStructure.NVIC_IRQChannel=USART3_IRQn;//特定的通道
	NVIC_InitStructure.NVIC_IRQChannelCmd=ENABLE;//通道使能
	NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority=2;//抢占式优先级为2
	NVIC_InitStructure.NVIC_IRQChannelSubPriority=1;//响应优先级为1
	NVIC_Init(&NVIC_InitStructure);
	
	USART_Cmd(USART1,ENABLE);//USART1使能打开
	USART_Cmd(USART3,ENABLE);//USART3使能打开
}

void USART1_IRQHandler(void)
{
	if(USART_GetITStatus(USART1,USART_IT_RXNE)==SET)//如果接受到
	{
		Sustainedmove=0;//持续运动关闭
		
		if(USART_ReceiveData(USART1)==0x29)//放松的趴下
		{
			Face_Mode=0;
			Face_Config();
			Action_Mode=0;
		}
		else if(USART_ReceiveData(USART1)==0x30)//蹲下
		{
			Face_Mode=1;
			Face_Config();
			Action_Mode=1;
		}
		else if(USART_ReceiveData(USART1)==0x31)//直立
		{
			Face_Mode=5;
			Face_Config();
			Action_Mode=2;

		}
		else if(USART_ReceiveData(USART1)==0x32)//趴下
		{
			Face_Mode=1;
			Face_Config();
			Action_Mode=3;
		}
		else if(USART_ReceiveData(USART1)==0x33)//前进
		{
			Face_Mode=2;
			Face_Config();
			Action_Mode=4;
		}
		else if(USART_ReceiveData(USART1)==0x34)//后退
		{
			Face_Mode=2;
			Face_Config();
			Action_Mode=5;
		}
		else if(USART_ReceiveData(USART1)==0x35)//左转
		{
			Face_Mode=2;
			Face_Config();
			Action_Mode=6;
		}
		else if(USART_ReceiveData(USART1)==0x36)//右转
		{
			Face_Mode=2;
			Face_Config();
			Action_Mode=7;
		}
		else if(USART_ReceiveData(USART1)==0x37)//摇摆
		{
			Face_Mode=4;
			Face_Config();
			Action_Mode=8;
		}
		else if(USART_ReceiveData(USART1)==0x38)//减少移动延迟，增加移动速度
		{
			if(SpeedDelay==120){Face_Mode=3;Face_Config();}
			if(SpeedDelay>100)
			SpeedDelay-=20;	
			else
			{
				Face_Mode=2;
				Face_Config();
				SpeedDelay=200;
			}

				
		}
		else if(USART_ReceiveData(USART1)==0x39)//减少摇摆延迟，增加摇摆速度
		{
			if(SwingDelay==4){Face_Mode=3;Face_Config();}
			if(SwingDelay>3)
			SwingDelay--;	
			else
			{	
				Face_Mode=4;
				Face_Config();
				SwingDelay=9;
			}
		}
		
		else if(USART_ReceiveData(USART1)==0x40)//摇尾巴
		{
			(WeiBa==0)?(WeiBa=1):(WeiBa=0);
			Face_Mode=1;
			Face_Config();
			Action_Mode=9;
		}
		
		else if(USART_ReceiveData(USART1)==0x41)//向前跳
		{
			Face_Mode=2;
			Face_Config();
			Action_Mode=10;
		}
		
		else if(USART_ReceiveData(USART1)==0x42)//向后跳
		{
			Face_Mode=2;
			Face_Config();
			Action_Mode=11;
		}
		else if(USART_ReceiveData(USART1)==0x43)//打招呼
		{
			Face_Mode=6;
			Face_Config();
			Action_Mode=13;
		}
		else if(USART_ReceiveData(USART1)==0x44)//开启灯光
		{
			AllLed=1;
		}
		else if(USART_ReceiveData(USART1)==0x45)//关闭灯光
		{
			AllLed=0;
		}
		else if(USART_ReceiveData(USART1)==0x46)//开启呼吸灯
		{
			BreatheLed=1;
		}
		else if(USART_ReceiveData(USART1)==0x47)//关闭呼吸灯
		{
			BreatheLed=0;
		}
		else if(USART_ReceiveData(USART1)==0x48)//伸懒腰
		{
			Face_Mode=6;
			Face_Config();
			Action_Mode=14;
		}
		else if(USART_ReceiveData(USART1)==0x49)//伸懒腰
		{
			Face_Mode=6;
			Face_Config();
			Action_Mode=15;
		}
		USART_ClearITPendingBit(USART1,USART_IT_RXNE);
	}
	
}

void USART3_IRQHandler(void)
{
	if(USART_GetITStatus(USART3,USART_IT_RXNE)==SET)//如果接受到
	{
		Sustainedmove=1;//持续运动开启
		
		if(USART_ReceiveData(USART3)==0x29)//放松的趴下
		{
			Face_Mode=0;
			Face_Config();
			Action_Mode=0;
		}
		else if(USART_ReceiveData(USART3)==0x30)//坐下
		{
			Face_Mode=1;
			Face_Config();
			Action_Mode=1;
		}
		else if(USART_ReceiveData(USART3)==0x31)//直立
		{
			Face_Mode=5;
			Face_Config();
			Action_Mode=2;

		}
		else if(USART_ReceiveData(USART3)==0x32)//趴下
		{
			Face_Mode=1;
			Face_Config();
			Action_Mode=3;
		}
		else if(USART_ReceiveData(USART3)==0x33)//前进
		{
			Face_Mode=2;
			Face_Config();
			Action_Mode=4;
		}
		else if(USART_ReceiveData(USART3)==0x34)//后退
		{
			Face_Mode=2;
			Face_Config();
			Action_Mode=5;
		}
		else if(USART_ReceiveData(USART3)==0x35)//左转
		{
			Face_Mode=2;
			Face_Config();
			Action_Mode=6;
		}
		else if(USART_ReceiveData(USART3)==0x36)//右转
		{
			Face_Mode=2;
			Face_Config();
			Action_Mode=7;
		}
		else if(USART_ReceiveData(USART3)==0x37)//摇摆
		{
			Face_Mode=4;
			Face_Config();
			Action_Mode=8;
		}
		else if(USART_ReceiveData(USART3)==0x38)//减少移动延迟，增加移动速度
		{
			if(SpeedDelay==120){Face_Mode=3;Face_Config();}
			if(SpeedDelay>100)
			SpeedDelay-=20;	
			else
			{
				Face_Mode=2;
				Face_Config();
				SpeedDelay=200;
			}

				
		}
		else if(USART_ReceiveData(USART3)==0x39)//减少摇摆延迟，增加摇摆速度
		{
			if(SwingDelay==4){Face_Mode=3;Face_Config();}
			if(SwingDelay>3)
			SwingDelay--;	
			else
			{	
				Face_Mode=4;
				Face_Config();
				SwingDelay=9;
			}
		}
		
		else if(USART_ReceiveData(USART3)==0x40)//摇尾巴
		{
			//(WeiBa==0)?(WeiBa=1):(WeiBa=0);
			Face_Mode=1;
			Face_Config();
			Action_Mode=9;
		}
		else if(USART_ReceiveData(USART3)==0x28)//摇尾巴停止
		{
			Face_Mode=1;
			Face_Config();
			Action_Mode=16;
		}
		
		else if(USART_ReceiveData(USART3)==0x41)//向前跳
		{
			Face_Mode=2;
			Face_Config();
			Action_Mode=10;
		}
		
		else if(USART_ReceiveData(USART3)==0x42)//向后跳
		{
			Face_Mode=2;
			Face_Config();
			Action_Mode=11;
		}
		else if(USART_ReceiveData(USART3)==0x43)//打招呼
		{
			Face_Mode=6;
			Face_Config();
			Action_Mode=13;
		}
		else if(USART_ReceiveData(USART3)==0x44)//开启灯光
		{
			AllLed=1;
		}
		else if(USART_ReceiveData(USART3)==0x45)//关闭灯光
		{
			AllLed=0;
		}
		else if(USART_ReceiveData(USART3)==0x46)//开启呼吸灯
		{
			BreatheLed=1;
		}
		else if(USART_ReceiveData(USART3)==0x47)//关闭呼吸灯
		{
			BreatheLed=0;
		}
		else if(USART_ReceiveData(USART3)==0x48)//伸懒腰
		{
			Face_Mode=6;
			Face_Config();
			Action_Mode=14;
		}
		else if(USART_ReceiveData(USART3)==0x49)//拉伸腿
		{
			Face_Mode=6;
			Face_Config();
			Action_Mode=15;
		}
		
		USART_ClearITPendingBit(USART3,USART_IT_RXNE);
	}
}
