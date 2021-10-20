import socket
import pickle, struct
import sys
from threading import Lock, Thread
import threading


class COMM:
    def __init__(self, host, port, user_id):
        self.host = host
        self.port = port
        self.id = user_id
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect((host,port))

    # the mess_type defines the content sent to server
    # -1 means start request
    # 0 means W
    # 1 means loss
    # 9 means straggler end connection
    # 10 means end connection
    def send2server(self,content,mess_type):
        data = pickle.dumps(content, protocol = 0)
        size = sys.getsizeof(data)

        header = struct.pack("i",size)
        u_id = struct.pack("i",self.id)
        mess_type = struct.pack("i",mess_type)

        self.client.sendall(header)
        self.client.sendall(u_id)
        self.client.sendall(mess_type)
        self.client.sendall(data)

    def recvfserver(self):
        header = self.client.recv(4)
        size = struct.unpack('i',header)

        recv_data = b""
        while sys.getsizeof(recv_data)<size[0]:
            recv_data += self.client.recv(size[0]-sys.getsizeof(recv_data))

        data = recv_data
        # print("the received data size is: {}, while the size of notice is: {}".format(sys.getsizeof(data),size[0]))
        data = pickle.loads(data)

        return data

    def recvOUF(self):
        #receive Omega from server
        header = self.client.recv(4)
        size = struct.unpack('i',header)

        recv_data = b""
        while sys.getsizeof(recv_data)<size[0]:
            recv_data += self.client.recv(size[0]-sys.getsizeof(recv_data))

        data = recv_data
        Omega = pickle.loads(data)


        #receive U from server
        header = self.client.recv(4)
        size = struct.unpack('i',header)

        recv_data = b""
        while sys.getsizeof(recv_data)<size[0]:
            recv_data += self.client.recv(size[0]-sys.getsizeof(recv_data))

        data = recv_data
        U = pickle.loads(data)


        #receive F from server
        header = self.client.recv(4)
        size = struct.unpack('i',header)

        recv_data = b""
        while sys.getsizeof(recv_data)<size[0]:
            recv_data += self.client.recv(size[0]-sys.getsizeof(recv_data))

        data = recv_data
        F = pickle.loads(data)
        # print(F)

        sig_stop = self.client.recv(4)
        sig_stop = struct.unpack('i',sig_stop)[0]

        return Omega, U, F, sig_stop


    def disconnect(self, type):
        if type==1:
            self.send2server('end',10)
        elif type==0:
            self.send2server('end',9)
        self.client.close()
