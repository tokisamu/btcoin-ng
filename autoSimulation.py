import networkx as nx
import numpy as np
import math
import subprocess
import socket
import os
import re
import sys
import threading
import time
import matplotlib.pyplot as plt
from functools import wraps
from pathlib import Path
import errno

HOME = f'{str(Path.home())}/regtest'
print(HOME)

# Decorator that adds identifying information to method calls
def identify(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        print(f'Node {self.nid} executing "{method.__name__}"')
        return method(self, *args, **kwargs)
    return wrapper


class Network:

    def __init__(self, n_nodes=6, topology='rand', connect_nodes=True):
        self.nid = "Master"
        self.n_nodes = n_nodes
        self.topology = topology
        self.network: nx.Graph = None
        self.connect_nodes = connect_nodes
        self.nodes = {}
        self.ports = [self._free_port() for _ in range(n_nodes*2)] # reserve two ports for each node
        try:
            self.setup()
        except Exception as e:
            print(f'ERROR! {e}')
            self.close()
    
    @identify
    def setup(self):
        if self.topology == 'rand':
            self.network = nx.erdos_renyi_graph(self.n_nodes, 0.5)
            while(not nx.is_connected(self.network)): # should not happen very often
                self.network = nx.erdos_renyi_graph(self.n_nodes, 0.5)
        elif self.topology == 'complete':
            self.network = nx.complete_graph(self.n_nodes)
        elif self.topology == 'ba':
            self.network = nx.barabasi_albert_graph(self.n_nodes, 8, seed=None)
        elif self.topology == 'grid':
            side = math.floor(math.sqrt(self.n_nodes))
            self.network = nx.grid_2d_graph(side, side)
            self.network = nx.convert_node_labels_to_integers(self.network, ordering = 'sorted')
        elif self.topology == 'ring':
            def ring_graph(n, k=1):
                G = nx.Graph()
                nxk = np.arange(0, n).repeat(k)
                src = nxk.reshape(n, k)
                dst = np.mod(np.tile(np.arange(0, k), n) + (nxk + 1), n).reshape((n, k))
                flat_pairs = np.dstack((src, dst)).flatten().tolist()
                edges = list(zip(flat_pairs[::2], flat_pairs[1::2]))
                G.add_edges_from(edges)
                return G
            self.network = ring_graph(self.n_nodes)
        else:
            print("Topology is not recognized!")
            self.close()
        nx.draw_networkx(self.network)
        plt.show()

        # create the actual network
        print("Setting up the network...")
        print("Starting up the nodes")
        self.create_all_nodes()
        if self.connect_nodes:
            print("Connecting the nodes")
            self.connect_all_nodes()
        print("Creating wallets")
        self.create_all_wallets()
        print("Done!")
    
    def create_all_nodes(self):
        import time
        for nid in self.network.nodes:
            self.nodes[nid] = Node(nid, self.network, self.ports.pop(), self.ports.pop())
            # time.sleep(1)

    def connect_all_nodes(self):
        for nid, node in self.nodes.items():
            for neighbor_id in self.network.neighbors(nid):
                node.adj_nodes.add((neighbor_id, self.nodes[neighbor_id].port))
            node.add_all_nodes()

    def create_all_wallets(self):
        for _, node in self.nodes.items():
            node.createwallet()
    
    # TODO print an overview table
    def print_table(self):
        pass

    @identify
    def close(self):
        # subprocess.run(['killall --regex bitcoin.*'], shell=True)
        print("Shutting network down...")
        for _, node in self.nodes.items():
            node.stop()
        exit()

    # TODO race conditions are unlikely but still possible; improve in the future
    def _free_port(self, port=1024, max_port=65535):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while port <= max_port:
            try:
                sock.bind(('', port))
                # sock.close()
                # reserve and return a bound socket to (hopefully) avoid race conditions
                # close and rebind when actually needed
                return sock
            except OSError:
                port += 1
        raise IOError('no free ports')

class Node:
    def __init__(self, nid, network, port, rpcport):
        self.nid = nid
        self.network: nx.Graph = network # TODO remove if not needed in the future
        self.adj_nodes = set() # set of (nid, port) of the adjacent nodes
        self.datadir = HOME + f'/Node{str(self.nid)}/'
        self._port = port
        self._rpcport = rpcport
        self._start()
            
    # RPC wrappers -------------------------------------------------------
    # -discover ?

    def addnode(self, nid, port, command='add'):
        """
        Add/remove a node
        https://developer.bitcoin.org/reference/rpc/addnode.html
        nid: id of the node
        port: port of the node
        command: add|remove|onetry
        """
        self._runcmd(self.cli_prefix + f'addnode "localhost:{port}" "{command}"')
        print(f'Node {self.nid} connected to Node {nid}!')
    
    @identify
    def stop(self):
        """
        Stop Bitcoin server.
        https://developer.bitcoin.org/reference/rpc/stop.html
        """
        self._runcmd(self.cli_prefix + f'stop')
    
    @identify
    def createwallet(self):
        """
        Creates and loads a new wallet.
        https://developer.bitcoin.org/reference/rpc/createwallet.html
        """
        wallet_name = f'wallet_node{self.nid}'
        if not os.path.exists(self.datadir + f'/regtest/wallets/{wallet_name}'):
            self._runcmd(self.cli_prefix + f'createwallet {wallet_name}')
        else:
            self._runcmd(self.cli_prefix + f'loadwallet {wallet_name}')
    
    
    # Helper methods -------------------------------------------------------
    
    @property
    def port(self):
        """
        Close the reserved socket and return its port so that it can be bound
        by the actual bitcoind process.
        """
        if isinstance(self._port, socket.socket):
            port = self._port.getsockname()[1]
            self._port.close()
            self._port = port
        return self._port

    @property
    def rpcport(self):
        """
        Close the reserved socket and return its port so that it can be bound
        by the actual bitcoind process.
        """
        if isinstance(self._rpcport, socket.socket):
            rpcport = self._rpcport.getsockname()[1]
            self._rpcport.close()
            self._rpcport = rpcport
        return self._rpcport

    def add_all_nodes(self):
        for nid, port in self.adj_nodes:
            self.addnode(nid, port)

    def _start(self):
        create_dir(self.datadir)
        self._runcmd(f'bitcoind -regtest -fallbackfee=0.00001 -server -daemon -debug -listen -port={self.port} -rpcport={self.rpcport} -datadir={self.datadir}')
        self.cli_prefix = f'bitcoin-cli -regtest -datadir={self.datadir} -rpcport={self.rpcport} '
    
    def _runcmd(self, cmd: str, suppress=False):
        try:
            result = subprocess.run(cmd, shell=True, check=True)
            if not suppress:
                print(result)
        except subprocess.CalledProcessError as e:
            print(f'ERROR in Node {self.nid}: {e.stderr}')
            # TODO graceful exit    

def create_dir(path):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as e:
            print(f'ERROR: {e}')
            if e.errno != errno.EEXIST:
                raise

    with open(f'{path}bitcoin.conf', "w") as f:
            f.write('rpcuser=user\nrpcpassword=pass\n')

def executeCommand(command,net):
    pat = re.compile('node {0,1}[0-9]+')
    nids = list(map(lambda x: x.replace('node','').strip(), pat.findall(command)))
    # get first node's cli_prefix
    command = re.sub(pat, str(net.nodes[int(nids[0])].cli_prefix.strip()), command, 1).strip()
    # get ports of all other nodes
    for nid in nids[1:]:
        command = re.sub(pat, str(net.nodes[int(nid)].port), command, 1).strip()
    print(f'Running: {command}')
    result = subprocess.run(command, shell=True, check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding="utf-8")
    stdout_as_str = result.stdout
    pat2 = re.compile('"address": "\w+"')
    return pat2.findall(stdout_as_str)[0]
def executeCommand2(command,net):
    pat = re.compile('node {0,1}[0-9]+')
    nids = list(map(lambda x: x.replace('node','').strip(), pat.findall(command)))
    # get first node's cli_prefix
    command = re.sub(pat, str(net.nodes[int(nids[0])].cli_prefix.strip()), command, 1).strip()
    # get ports of all other nodes
    for nid in nids[1:]:
        command = re.sub(pat, str(net.nodes[int(nid)].port), command, 1).strip()
    print(f'Running: {command}')
    result = subprocess.run(command, shell=True, check=True)
    return 1
# TODO create argparser later on
def main():
    """
    Create and manage a local network of bitcoin nodes.
    ----
    n_nodes: number of nodes in the network
    topology: rand|grid|complete|ring
    """
    """
    node0 -generate 1
    ...
    noden -generate 1
    node(n/2) -generate 101
    
    node(n/2) generateKeyblock address publickey

    nodei sendtoaddress "random" 0.0001
    repeat 10times
    node(n/2) -generateMicroblock address publickey private
    repeat 100times
    """
    commands = []
    address = []
    net = Network(n_nodes=100, topology='ba', connect_nodes=True)
    centerNode = round(net.n_nodes/2)
    for i in range(0,net.n_nodes):
        command =("node"+str(i)+" -generate 1");
        result = executeCommand(command,net)
        address.append(result[12:-1])
        time.sleep(1)
        #result = executeCommand(command,net)
        #result = executeCommand(command,net)
    command =("node"+str(centerNode)+" -generate 101");
    result = executeCommand2(command,net)
    time.sleep(2)
    for i in range(0,net.n_nodes):
        command =("node"+str(i)+" -generate 1");
        result = executeCommand(command,net)
        address.append(result[12:-1])
        time.sleep(0.5)
        result = executeCommand(command,net)
        time.sleep(0.5)
        #result = executeCommand(command,net)
    command =("node"+str(centerNode)+" -generate 101");
    result = executeCommand2(command,net)
    time.sleep(2)
    command =("node"+str(centerNode)+" generateKeyblock 1  "+address[centerNode]+" 998");
    result = executeCommand2(command,net)
    cnt = 10
    while(cnt):
        for repeat in range(0,1):
            time.sleep(1)
            for i in range(0,net.n_nodes):
                if i==centerNode:
                    continue;
                target = (i+centerNode)%net.n_nodes;
                if target==centerNode:
                    target+=1;
                command =("node"+str(i)+" sendtoaddress "+address[i]+" 0.0001");
                result = executeCommand2(command,net)
        command =("node"+str(centerNode)+" generateMicroblock 1  "+address[centerNode]+" 998 123");
        result = executeCommand2(command,net)
        for i in range(0,10):
            command =("node"+str(i)+" getwalletinfo");
            result = executeCommand2(command,net)
        cnt-=1
        #print(cnt)
    for i in range(0,10):
        command =("node"+str(centerNode)+" generateMicroblock 1  "+address[centerNode]+" 998 123");
        result = executeCommand2(command,net)
        time.sleep(3)
    '''
    for i in range(0,net.n_nodes):
        command =("node"+str(i)+" -generate 1");
        result = executeCommand(command,net)
        address.append(result[12:-1])
        time.sleep(0.5)
        #result = executeCommand(command,net)
    '''
    print("finished")
    net.close()


if __name__ == '__main__':
    main()