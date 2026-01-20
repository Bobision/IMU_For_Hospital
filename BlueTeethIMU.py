import serial
import time


def combine_bytes(data):
    # 获取高位和低位数据
    low_byte = data[0]
    high_byte = data[1]

    # 将高位和低位字节转换为整数
    low_value = int.from_bytes(low_byte, byteorder='little')
    high_value = int.from_bytes(high_byte, byteorder='little')

    # 将高位数据左移8位，并与低位数据组合
    result = high_value << 8 | low_value

    # 如果结果是一个负数，则进行符号位扩展
    if result >= 32768:
        result -= 65536  # 用65536减去结果，相当于在最高位添加符号位

    return result


def cal_data(data, para):
    cal_list = [0.0, 0.0, 0.0]

    cal_list[0] = combine_bytes(data[0:2]) * para
    cal_list[1] = combine_bytes(data[2:4]) * para
    cal_list[2] = combine_bytes(data[4:6]) * para

    return cal_list


class imu:
    def __init__(self, ser):
        self.ser = ser
        self.read_flag = 0
        self.read_count = 0
        self.data_list = []

        # 记录加速度
        self.data_acc02 = [0.0, 0.0, 0.0]
        self.data_acc03 = [0.0, 0.0, 0.0]
        self.data_acc04 = [0.0, 0.0, 0.0]
        self.data_acc05 = [0.0, 0.0, 0.0]
        self.data_acc06 = [0.0, 0.0, 0.0]
        self.data_acc07 = [0.0, 0.0, 0.0]
        self.data_acc08 = [0.0, 0.0, 0.0]
        # 记录角速度
        self.data_vel02 = [0.0, 0.0, 0.0]
        self.data_vel03 = [0.0, 0.0, 0.0]
        self.data_vel04 = [0.0, 0.0, 0.0]
        self.data_vel05 = [0.0, 0.0, 0.0]
        self.data_vel06 = [0.0, 0.0, 0.0]
        self.data_vel07 = [0.0, 0.0, 0.0]
        self.data_vel08 = [0.0, 0.0, 0.0]
        # 记录角度
        self.data_ang02 = [0.0, 0.0, 0.0]
        self.data_ang03 = [0.0, 0.0, 0.0]
        self.data_ang04 = [0.0, 0.0, 0.0]
        self.data_ang05 = [0.0, 0.0, 0.0]
        self.data_ang06 = [0.0, 0.0, 0.0]
        self.data_ang07 = [0.0, 0.0, 0.0]
        self.data_ang08 = [0.0, 0.0, 0.0]
        # 记录电池电量
        self.battery02 = 0
        self.battery03 = 0
        self.battery04 = 0
        self.battery05 = 0
        self.battery06 = 0
        self.battery07 = 0
        self.battery08 = 0

        # 将变量存储到字典中
        self.data_acc = {
            2: self.data_acc02,
            3: self.data_acc03,
            4: self.data_acc04,
            5: self.data_acc05,
            6: self.data_acc06,
            7: self.data_acc07,
            8: self.data_acc08,
        }
        self.data_vel = {
            2: self.data_vel02,
            3: self.data_vel03,
            4: self.data_vel04,
            5: self.data_vel05,
            6: self.data_vel06,
            7: self.data_vel07,
            8: self.data_vel08
        }
        self.data_ang = {
            2: self.data_ang02,
            3: self.data_ang03,
            4: self.data_ang04,
            5: self.data_ang05,
            6: self.data_ang06,
            7: self.data_ang07,
            8: self.data_ang08
        }
        self.battery = {
            2: self.battery02,
            3: self.battery03,
            4: self.battery04,
            5: self.battery05,
            6: self.battery06,
            7: self.battery07,
            8: self.battery08
        }

        # 设备号列表
        self.device = [b'\x02', b'\x03', b'\x04', b'\x05', b'\x06', b'\x07', b'\x08']
        self.device_number = 0

        # 常数计算
        self.acc = 16 * 9.81 / 32768
        self.vel = 2000 / 32768
        self.ang = 180 / 32768

        self.seq = {i: 0 for i in range(2, 9)}

    def read(self):
        """
        快速读取：一次性取走串口缓冲区的所有字节，并逐字节喂给状态机。
        外部调用频率可降低到 0.5~2 ms 一次；即使 UI/存盘忙，也不易积压。
        """
        n = getattr(self.ser, "in_waiting", 0)
        if n and n > 0:
            buf = self.ser.read(n)
        else:
            # 没有积压就读1字节，保持兼容（注意给串口设置较小的 timeout）
            buf = self.ser.read(1)
        for b in buf:
        # print(byte)  # 输出字节的可读形式
        # print(byte == b'U')  # 对比与字符 'U' 的字节形式
        # print(byte == b'\x55')  # 对比与十六进制 0x55 的字节形式
        # print(byte[0])  # 获取第一个字节的数值（十进制）
            self.process_data(bytes([b]))

    def process_data(self, byte):
        if self.read_flag == 4:
            self.read_flag = 0
            return self.process_data(byte)  # 关键：复用同一字节作为新帧的起点
        if self.read_flag == 0:  # 初始状态
            if byte in self.device:  # 读到设备号
                self.device_number = int.from_bytes(byte, byteorder='big')
                # print(f"读到设备号 {self.device_number}")
                self.read_flag = 1
                return None
            else:
                self.read_flag = 0
        if self.read_flag == 1:  # 已经读到设备号
            if byte == b'\x55':  # 读到数据包头
                # print(f"读到设备号 {self.device_number}的数据111")
                self.read_flag = 2
                return None
            else:
                self.read_flag = 0
        elif self.read_flag == 2:  # 已经读到数据包头
            if byte == b'\x61':  # 读到标志位
                # print(f"读到设备号 {self.device_number}的数据222")
                self.read_flag = 3
                self.data_list = []
                self.read_count = 0
                return None
            else:
                self.read_flag = 0
        elif self.read_flag == 3:  # 准备开始读取数据内容（加速度、角速度、角度）
            if self.read_count <= 24:  # 向后读取26位
                self.data_list.append(byte)
                self.read_count = self.read_count + 1
                return None
            elif self.read_count == 25:
                self.data_list.append(byte)
                # print(self.data_list)
                # print(cal_data(self.data_list[0:6], self.acc))
                self.data_acc[self.device_number] = cal_data(self.data_list[0:6], self.acc)
                self.data_vel[self.device_number] = cal_data(self.data_list[6:12], self.vel)
                self.data_ang[self.device_number] = cal_data(self.data_list[18:24], self.ang)
                # self.battery[self.device_number] = combine_bytes(self.data_list[24:26])
                self.battery[self.device_number] = round(get_battery_percentage(combine_bytes(self.data_list[24:26])), 2)
                # print(self.data_ang[self.device_number])
                #自增标志位
                #self.seq[self.device_number] = (self.seq[self.device_number] + 1) & 0xFFFFFFFF
                self.read_flag = 4
            else:
                self.read_flag = 0
        else:
            self.read_flag = 0


def get_battery_percentage(value):
    if value > 396:
        return 100
    elif 393 <= value <= 396:
        return 90 + (value - 393) * 10 / 3
    elif 387 <= value < 393:
        return 75 + (value - 387) * 15 / 6
    elif 382 <= value < 387:
        return 60 + (value - 382) * 15 / 5
    elif 379 <= value < 382:
        return 50 + (value - 379) * 10 / 3
    elif 377 <= value < 379:
        return 40 + (value - 377) * 10 / 2
    elif 373 <= value < 377:
        return 30 + (value - 373) * 10 / 3
    elif 370 <= value < 373:
        return 20 + (value - 370) * 10 / 3
    elif 368 <= value < 370:
        return 15 + (value - 368) * 5 / 2
    elif 350 <= value < 368:
        return 10 + (value - 350) * 10 / 18
    elif 340 <= value < 350:
        return 5 + (value - 340) * 5 / 10
    else:
        return 0  # 如果读到的值小于340，电量为0%
