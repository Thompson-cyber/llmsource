import socket
import struct


# TCP server commands
TCP_CMD_PING = 0x01  
TCP_CMD_ON = 0x02
TCP_CMD_OFF = 0x03
TCP_CMD_READ_RAM = 0x10
TCP_CMD_READ_EEPROM = 0x11
TCP_CMD_WRITE_RAM  = 0x20 
TCP_CMD_WRITE_EEPROM = 0x21
TCP_CMD_IR_CMD = 0x30
# TCP server errors
TCP_ERR_GENERAL = 0xFF
TCP_ERR_LOCK = 0xFE
TCP_ERR_CHECKSUM = 0xFD
TCP_ERR_NOCMD = 0xFC
# Stove commands
STOVE_ADDR_AMBIENT_TEMP = 0x01
STOVE_ADDR_STATE = 0x21
STOVE_ADDR_FUMES_TEMP = 0x3E
STOVE_ADDR_FUMES_SPEED = 0x37
STOVE_ADDR_POWER_RAM = 0x19
STOVE_ADDR_POWER_EEPROM = 0x7F
STOVE_ADDR_IRCOMMAND = 0x58
STOVE_ADDR_THERMOSTAT = 0x7D
# Stove states
STOVE_STATE_TURN_ON = 0x01
STOVE_STATE_TURN_OFF = 0x06
STOVE_STATE_FORCE_OFF = 0x00
# ircommands
STOVE_IR_POWER = 0x5A



# this code was generated by chatgpt from input src/crc8.c
def crc8(data: bytes) -> int:
    polyval = 0xEB
    init = 0xFF
    crc = init
    for b in data:
        crc ^= b
        for i in range(8):
            msb = crc & 0x80
            crc <<= 1
            if msb:
                crc ^= polyval
    return crc & 0xff



class MicronovaClient:
    def __init__(self, hostname, port):
        self.hostname = hostname
        self.port = port
        self.create_socket()
        return

    
    def create_socket(self):
        self.socket = socket.socket(
            socket.AF_INET, 
            socket.SOCK_STREAM
        )
    

    def close(self):
        self.socket.close()


    def reconnect(self):
        self.close()
        self.create_socket()
        self.connect()


    def connect(self):
        self.socket.connect( (self.hostname, self.port) )
    

    def ping(self):
        self.send( struct.pack("=B", TCP_CMD_PING) )
        res = self.recive()
        if res[0]==TCP_CMD_PING:
            return True
        else:
            return False
    

    def send(self, message):
        message+=bytes( [crc8(message) ])
        self.socket.sendall(message)
    
    
    def recive(self):
        # TODO: checksum error check & chcksum validy check
        return self.socket.recv(10)
    
    
    def read(self, location, address):
        cmd = struct.pack("=BB",
            location,
            address
        )
        self.send(cmd)
        return self.recive()
    

    def read_ram(self, addr):
        return self.read(
            TCP_CMD_READ_RAM,
            addr
        )


    def read_eeprom(self, addr):
        return self.read(
            TCP_CMD_READ_EEPROM,
            addr
        )


    def write(self, location, addr, data, read_resp=False):
        response_request = 0x01 if read_resp else 0x00
        cmd = struct.pack("=BBBB",
            location,
            addr,
            data,
            response_request
        )
        self.send(cmd)
        return self.recive()


    def write_ram(self, addr, data, read_resp=False):
        return self.write(
            TCP_CMD_WRITE_RAM,
            addr,
            data,
            read_resp
        )


    def write_eeprom(self, addr, data, read_resp=False):
        return self.write(
            TCP_CMD_WRITE_EEPROM,
            addr,
            data,
            read_resp
        )


    def simulate_infrared(self, command, repetions):
        cmd = struct.pack("=BBB",
            TCP_CMD_IR_CMD,
            command,
            repetions
        )
        self.send(cmd)
        return self.recive()

    # helpers
    def convert_number_for_clock(self, number):
        return int(str(number),16)

    # abstraction functions
    def get_stove_state(self):
        r = self.read_ram( STOVE_ADDR_STATE )
        return r[1]

    def on(self):
        self.write_ram( STOVE_ADDR_STATE, STOVE_STATE_TURN_ON )

    def off(self):
        self.simulate_infrared( STOVE_IR_POWER, 10 )

    # WARNING: DO NOT USE IF NOT 100% NEEDED, 
    # THIS CAN POTENTIALLY BREAK THE STOVE IF ITS RUNNING!!
    def force_off(self):
        self.write_ram( STOVE_ADDR_STATE, STOVE_STATE_FORCE_OFF )

    def get_ambient_temp(self):
        r = self.read_ram( STOVE_ADDR_AMBIENT_TEMP )
        return r[1] / 2

    def get_fumes_temp(self):
        r = self.read_ram( STOVE_ADDR_FUMES_TEMP )
        return r[1]
    
    def get_fumes_fan_speed(self):
        r = self.read_ram( STOVE_ADDR_FUMES_SPEED )
        return r[1]*10

    def set_power(self, power):
        if power > 0: 
            power = 0
        if power > 4:
            power = 4
        self.write_eeprom(STOVE_ADDR_POWER_EEPROM, power)
        self.write_ram(STOVE_ADDR_POWER_RAM, power)

    def get_power(self):
        r = self.read_eeprom( STOVE_ADDR_POWER_EEPROM )
        return r[1]

    def get_air_out_temp(self):
        r = self.read_ram(0x02)
        return r[1]

    def get_air_out_fans_speed(self):
        r = self.read_ram(0x0A)
        return r[2]
    
    def set_theromstat_temp(self, temp):
        if temp<0:
            temp = 0
        if temp > 32:
            temp=32
        self.write_eeprom( STOVE_ADDR_THERMOSTAT, temp )
