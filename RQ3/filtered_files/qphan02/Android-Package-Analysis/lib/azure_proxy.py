import __root__
from config import azure_config
from lib import Proxy

from azure.identity import ClientSecretCredential

from azure.mgmt.network import NetworkManagementClient
from azure.mgmt.network.models import NetworkInterface, Subnet, IPConfiguration, PublicIPAddress

from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.compute.models import VirtualMachine

from typing import Iterable, Optional, List, Any

from icecream import ic
from pprint import pprint

from copy import deepcopy
import random

import logging
# logging.basicConfig(filename='azure_proxy.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

class AzureProxy(Proxy):
    """
    Handle Azure Proxy
    """
    
    PROXY_PORT: int = 8888
    
    ## Azure variables
    __network_client: NetworkManagementClient
    __compute_client: ComputeManagementClient
    
    ## State of Azure
    __virtual_machines: Iterable[VirtualMachine]
    __network_interfaces: Iterable[NetworkInterface]
    __ip_configs: Iterable[IPConfiguration]
    __subnets: Iterable[Subnet]
    __ip_addrs: Iterable[PublicIPAddress]
    __nsg_id: str
    __ip_nic_map: dict[str: NetworkInterface]
    __reserved_ip_addresses: list[PublicIPAddress]
    
    
    def __init__(self, num_active_proxies: int,  num_reserved_proxies: int, start_vm: Optional[bool] = False) -> None:
        ## initialize parent
        super().__init__(
            num_active_proxies=num_active_proxies, 
            num_reserved_proxies=num_reserved_proxies,
        )
        
        self._init_mgmt_objects()
        self._get_state()
        
            
        ## get list of resevered ip addresses
        self.__reserved_ip_addresses = self._find_unassigned_ip_addresses()
        reseved_ips_deficit = self.num_reserved_proxies - len(self.__reserved_ip_addresses)
        
        ## create more public ip addresses until it meets the requirement
        while (reseved_ips_deficit > 0):
            self._create_public_ip_address(wait=True)
            reseved_ips_deficit -= 1
            
            
        ## when len(vm) ≤ num_active_proxies --> create more vm
        vms = list(deepcopy(self.__virtual_machines))
        vms_count = len(vms)
        if vms_count < self.num_active_proxies:
            ## TODO: add more VMs
            pass
        elif vms_count > self.num_active_proxies:
            ## TODO: reduce number of VMs
            pass
        
        ## start up VMs
        if start_vm is True:
            self.start()
                

    def get_proxies(self) -> list[str]:
        active_ip_addrs = self.__ip_nic_map.keys()
        proxies = [f'{ip}:{self.PROXY_PORT}' for ip in active_ip_addrs]
        return proxies


    def replace_proxy(self, proxy: str) -> str:
        """
        """
        
        ## get public ip from proxy
        public_ip_addr = proxy.split(':',1)[0]
        
        ## retrieve corresponding NetworkInterface
        nic: NetworkInterface = self.__ip_nic_map[public_ip_addr]
        
        ## remove it from ip_nic_map
        assert public_ip_addr in self.__ip_nic_map, f'{public_ip_addr} does not exist!'
        del self.__ip_nic_map[public_ip_addr]
        
        ## pop one public ip from list of reserve
        public_ip: PublicIPAddress = self.__reserved_ip_addresses.pop(0)
        
        ## create a new reserved public ip address
        self._create_public_ip_address(wait=False)
        
        ## associate new public ip to NetworkInterface
        nic, old_public_ip_name = self._associate_public_ip_to_nic(
            nic=nic,
            public_ip=public_ip,
            wait=True,
        )
        
        ## add new ip to ip_nic_map
        self.__ip_nic_map[public_ip.ip_address] = nic
        
        ## remove the old public ip
        self._remove_public_ip_address(old_public_ip_name, wait = False)
        
        ## return the new public ip address
        return f"{public_ip.ip_address}:{self.PROXY_PORT}"
    
    
    def start(self) -> None:
        """
        start up Azure VMs to run proxies
        """
        
        vm_objs = []
        for vm in deepcopy(self.__virtual_machines):
            vm_name = self._get_name_from_id(vm.id)
            vm_obj = self._start_vm(vm_name, wait = False)
            vm_objs.append(vm_obj)
            
        for vm_obj in vm_objs:
            vm_obj.wait()
            
        self.proxy_server_is_ready = True
            
        logging.info("All Azure VMs are running!")
    
    
    def stop(self) -> None:
        """
        after no longer use azure,
        close everything to save cost
        """
        
        self.proxy_server_is_ready = False
        
        ## deallocate VMs (to save cost)
        vm_objs = []
        for vm in deepcopy(self.__virtual_machines):
            vm_name = self._get_name_from_id(vm.id)
            vm_obj = self._deallocate_vm(vm_name, wait = False)
            vm_objs.append(vm_obj)
            
        for vm_obj in vm_objs:
            vm_obj.wait()
            
        logging.info("All Azure VMs have been deallocated!")
            
    
    """Private methods"""
    
    
    def _init_mgmt_objects(self) -> None:
        ## Create client credential object
        self.__azure_credential = ClientSecretCredential(
            tenant_id=azure_config.TENANT_ID,
            client_id=azure_config.CLIENT_ID,
            client_secret=azure_config.SECRET_KEY,
        )

        # self.__azure_credential = DefaultAzureCredential()
        
        ## Create the network management client object
        self.__network_client = NetworkManagementClient(
            credential=self.__azure_credential,
            subscription_id=azure_config.SUBSCRIPTION_ID,
        )
        
        ## Create the compute management client object object
        self.__compute_client = ComputeManagementClient(
            credential=self.__azure_credential,
            subscription_id=azure_config.SUBSCRIPTION_ID,
        )
        
        ## initialize the nsg_id
        self.__nsg_id = '/subscriptions/' + azure_config.SUBSCRIPTION_ID + '/resourceGroups/' + azure_config.RESOURCE_GROUP + '/providers/Microsoft.Network/networkSecurityGroups/' + azure_config.NSG_NAME
        
    
    def _get_state(self) -> None:
        ## Get list of VMs
        self.__virtual_machines = self.__compute_client.virtual_machines.list(azure_config.RESOURCE_GROUP) 
        
        ## Get list of Network Interfaces
        self.__network_interfaces = self.__network_client.network_interfaces.list(azure_config.RESOURCE_GROUP)
        
        ## Get list of IP Configurations
        self.__ip_configs = [ipconfig for ni in deepcopy(self.__network_interfaces) for ipconfig in ni.ip_configurations]

        ## Get list of subnets
        self.__subnets = self.__network_client.subnets.list(
            resource_group_name=azure_config.RESOURCE_GROUP,
            virtual_network_name=azure_config.VIRTUAL_NETWORK,
        )
        
        ## Get list of IP Addresses
        self.__ip_addrs = self.__network_client.public_ip_addresses.list_all()
        
        ## update ip_nic_map
        self.__ip_nic_map = dict()
        for nic in deepcopy(self.__network_interfaces):
            public_ip_name: str = nic.ip_configurations[0].public_ip_address.id.rsplit('/',1)[-1]
            public_ip = self.__network_client.public_ip_addresses.get(
                azure_config.RESOURCE_GROUP,
                public_ip_address_name=public_ip_name,
            )
            public_ip_addr = public_ip.ip_address
            self.__ip_nic_map[public_ip_addr] = nic
    
    
    def _get_name_from_id(self, id: str) -> str:
        name = id.rsplit('/',1)[-1]
        return name
    
    
    def _create_vm(self) -> VirtualMachine:
        """
        create a virtual machine
        """
        # Create a new virtual machine configuration
        ## TODO: this function is generated by Copilot, and it's possibly not working
        vm_config = self.__compute_client.virtual_machines.begin_create_or_update(
            azure_config.RESOURCE_GROUP,
            azure_config.VM_NAME,
            {
                "location": azure_config.LOCATION,
                "hardware_profile": {
                    "vm_size": azure_config.VM_SIZE
                },
                "storage_profile": {
                    "image_reference": {
                        "publisher": azure_config.IMAGE_PUBLISHER,
                        "offer": azure_config.IMAGE_OFFER,
                        "sku": azure_config.IMAGE_SKU,
                        "version": azure_config.IMAGE_VERSION
                    },
                    "os_disk": {
                        "name": azure_config.OS_DISK_NAME,
                        "create_option": "fromImage",
                        "caching": "ReadWrite",
                        "managed_disk": {
                            "storage_account_type": "Standard_LRS"
                        }
                    },
                    "data_disks": []
                },
                "os_profile": {
                    "computer_name": azure_config.VM_NAME,
                    "admin_username": azure_config.ADMIN_USERNAME,
                    "admin_password": azure_config.ADMIN_PASSWORD,
                    "linux_configuration": {
                        "disable_password_authentication": False
                    }
                },
                "network_profile": {
                    "network_interfaces": [
                        {
                            "id": self.__nic.id,
                            "properties": {
                                "primary": True
                            }
                        }
                    ]
                }
            }
        ).result()

        # Start the virtual machine
        self.__compute_client.virtual_machines.begin_start(
            azure_config.RESOURCE_GROUP,
            azure_config.VM_NAME
        ).wait()

        return vm_config
        
        
    def _destroy_vm(self, vm_name: str, wait: Optional[bool] = True) -> Any:
        """
        destroy a virtual machine
        """
        r = self.__compute_client.virtual_machines.begin_delete(
            resource_group_name=azure_config.RESOURCE_GROUP, 
            vm_name=vm_name,
        )
        
        if wait is True:
            r.wait()
            
        return r
        
    
    
    def _deallocate_vm(self, vm_name: str, wait: Optional[bool] = True) -> Any:
        """
        """
        r = self.__compute_client.virtual_machines.begin_deallocate(
            resource_group_name=azure_config.RESOURCE_GROUP, 
            vm_name=vm_name,
        )
        
        if wait is True:
            r.wait()
            
        return r
    
    
    def _start_vm(self, vm_name: str, wait: Optional[bool] = True) -> Any:
        """
        """
        r = self.__compute_client.virtual_machines.begin_start(
            resource_group_name=azure_config.RESOURCE_GROUP, 
            vm_name=vm_name,
        )
        
        if wait is True:
            r.wait()
            
        return r
    
    
    def _get_ip_from_vm(self, vm_name: str, get_str: Optional[bool] = True) -> PublicIPAddress | str:
        """
        """
        ## Get the network interface associated with the VM
        vm: VirtualMachine = self.__compute_client.virtual_machines.get(
                resource_group_name=azure_config.RESOURCE_GROUP,
                vm_name=vm_name,
        )
        nic_name: str = vm.network_profile.network_interfaces[0].id.rsplit('/',1)[-1]

        ## Get the public IP address associated with the network interface
        nic: NetworkInterface = self.__network_client.network_interfaces.get(
            resource_group_name=azure_config.RESOURCE_GROUP, 
            network_interface_name=nic_name,
        )
        public_ip_name:str = nic.ip_configurations[0].public_ip_address.id.rsplit('/',1)[-1]

        ## Get the public IP address
        public_ip = self.__network_client.public_ip_addresses.get(
            resource_group_name=azure_config.RESOURCE_GROUP, 
            public_ip_address_name=public_ip_name,
        )
        
        ## return publi IP object if get_str is False
        if get_str is False:
            return public_ip
        
        public_ip_address: str = public_ip.ip_address
        return public_ip_address

    
    def _add_network_interface_to_vm(self, nic: NetworkInterface | str, vm: VirtualMachine | str):
        """
        """
        if isinstance(nic, str) is True:
            nic: NetworkInterface = self.__network_client.network_interfaces.get(
                resource_group_name=azure_config.RESOURCE_GROUP,
                network_interface_name=nic,
            )
            
        if isinstance(vm, str) is True:
            vm: VirtualMachine = self.__compute_client.virtual_machines.get(
                resource_group_name=azure_config.RESOURCE_GROUP,
                vm_name=vm,
            )
                        
        ## Add the NIC to the virtual machine's network interface list
        network_interfaces = [{"id": _nic.id, "primary": False} for _nic in vm.network_profile.network_interfaces]
        network_interfaces.insert(0, {"id":nic.id, "primary": True})
                            
        logging.INFO(f"Shutting down VM {vm.name}")
        
        
        ## Need to stop (deallocated) the virtual machine before update number of nic to it
        self.__compute_client.virtual_machines.begin_deallocate(
            resource_group_name=azure_config.RESOURCE_GROUP, 
            vm_name=vm.name,
        ).wait()
        
        logging.INFO(f"Updating down VM {vm.name}")
                
        ret = self.__compute_client.virtual_machines.begin_create_or_update(
            resource_group_name=azure_config.RESOURCE_GROUP, 
            vm_name=vm.name,
            parameters={
                "location": azure_config.LOCATION,
                "network_profile": {
                    "network_interfaces": network_interfaces
                },
            }
        )
                
        logging.INFO(f"Starting VM {vm.name}")
        
        ## start up the VM again
        self.__compute_client.virtual_machines.begin_start(
            resource_group_name=azure_config.RESOURCE_GROUP, 
            vm_name=vm.name,
        ).wait()
        
        logging.INFO(f"Done: {vm.name}")
        
        return ret
    
    
    def _create_public_ip_address(self, name: Optional[str] = None, wait: Optional[bool] = True) -> PublicIPAddress:
        """
        """
        if name is None:
            name = 'public_ip_address_' + self._generate_random_hash()
            name = name[:79] + '_'
            
        r = self.__network_client.public_ip_addresses.begin_create_or_update(
            resource_group_name=azure_config.RESOURCE_GROUP,
            public_ip_address_name=name,
            parameters={
                "location": azure_config.LOCATION, 
                'sku': {'name': 'standard'},
                'public_ip_allocation_method':'static',
                'public_ip_address_version':'ipv4'
            }
        )
        
        if wait is True:
            r.wait()
            
        new_public_ip = r.result()
        self.__reserved_ip_addresses.append(new_public_ip)
            
        return new_public_ip
    
    
    def _remove_public_ip_address(self, name: str, wait: Optional[bool] = False) -> None:
        """
        """
        assert isinstance(name, str) is True, f'name needs to be a string, not f{type(name)}'
        
        r = self.__network_client.public_ip_addresses.begin_delete(
            resource_group_name=azure_config.RESOURCE_GROUP,
            public_ip_address_name=name,
        )
        
        if wait is True:
            r.wait()
        
        return r.result()
    
    
    def _disassociate_public_ip_from_nic(self, nic: NetworkInterface | str, wait: Optional[bool] = True) -> NetworkInterface:
        """
        """
        if isinstance(nic, str) is True:
            nic: NetworkInterface = self.__network_client.network_interfaces.get(
                azure_config.RESOURCE_GROUP,
                network_interface_name=nic,
            )
        
        ipconfig: IPConfiguration = nic.ip_configurations[0]
        ipconfig.public_ip_address = None
                
        params = {
            'location': nic.location, 
            'ip_configurations': [ipconfig],
            'network_security_group': nic.network_security_group,
        }
            
        ## Update the NIC with the modified IP configuration
        r = self.__network_client.network_interfaces.begin_create_or_update(
            azure_config.RESOURCE_GROUP, 
            network_interface_name=nic.name, 
            parameters=params,
        )
        
        if wait is True:
            r.wait()
            
        return r.result()
    
    
    def _associate_public_ip_to_nic(
        self, 
        nic: NetworkInterface | str, 
        public_ip: PublicIPAddress | str, 
        wait: Optional[bool] = True
    ) -> tuple[NetworkInterface, str]:
        """
        """
        if isinstance(nic, str) is True:
            nic: NetworkInterface = self.__network_client.network_interfaces.get(
                azure_config.RESOURCE_GROUP,
                network_interface_name=nic,
            )
            
        if isinstance(public_ip, str) is True:
            public_ip: NetworkInterface = self.__network_client.public_ip_addresses.get(
                azure_config.RESOURCE_GROUP,
                public_ip_address_name=public_ip,
            )
            
        ipconfig: IPConfiguration = nic.ip_configurations[0]
        
        ## get old public ip name
        old_public_ip_name: str = ipconfig.public_ip_address.id.rsplit('/',1)[-1]
        
        ## assign new public ip to nic
        ipconfig.public_ip_address = {
            'id': public_ip.id,
        }
                
        params = {
            'location': nic.location, 
            'ip_configurations': [ipconfig],
            'network_security_group': nic.network_security_group,
        }
            
        ## Update the NIC with the modified IP configuration
        r = self.__network_client.network_interfaces.begin_create_or_update(
            azure_config.RESOURCE_GROUP, 
            network_interface_name=nic.name, 
            parameters=params,
        )
            
        if wait is True:
            r.wait()
            
        return r.result(), old_public_ip_name
    
    
    def _create_ip_config(
        self, 
        name: Optional[str] = None,
        subnet: Optional[Subnet] = None,
        public_ip_address: Optional[PublicIPAddress] = None,
    ) -> IPConfiguration:
        """
        """
        if name is None:
            name = 'ipconfig_' + self._generate_random_hash()

        if subnet is None:
            subnet = random.choice(list(deepcopy(self.__subnets)))
            
        ## Attemp to assign one of unassigned IP Addresses to public_ip_address
        if public_ip_address is True:
            unassigned_ip_addresses = self._find_unassigned_ip_addresses()
            if len(unassigned_ip_addresses) > 0:
                public_ip_address = unassigned_ip_addresses[0]
                
        ipconfg = IPConfiguration(
            name=name,
            subnet=subnet,
            public_ip_address=public_ip_address,
        )
        
        return ipconfg

    
    def _create_network_interface(
        self,
        name: Optional[str] = None,
        ip_config: Optional[IPConfiguration] = None,
    ) -> NetworkInterface:
        """
        """
        if isinstance(name, str) is False:
            name = 'network_interface_' + self._generate_random_hash()
                    
        if isinstance(ip_config, IPConfiguration) is False:
            ip_config = self._create_ip_config()

        params: dict = {
            'location': azure_config.LOCATION, 
            'ip_configurations': [ip_config],
            'network_security_group': {
                'id': self.__nsg_id,
            }
        }
        
        network_interface =  self.__network_client.network_interfaces.begin_create_or_update(
            resource_group_name=azure_config.RESOURCE_GROUP,
            network_interface_name=name,
            parameters=params,
        )
        network_interface.wait()
        
        return network_interface
    
 
    def _find_unassigned_ip_addresses(self) -> List[PublicIPAddress]:
        """Get a list of reserved public IP addresses"""
        self._get_state()
        ip_configs = deepcopy(self.__ip_configs)
        ip_addrs = deepcopy(self.__ip_addrs)
        assigned_ip_addresses: set = {ip_config.public_ip_address.id.rsplit('/')[-1] for ip_config in ip_configs if ip_config.public_ip_address is not None}
        all_ip_addresses: dict = {ip_addr.name: ip_addr for ip_addr in ip_addrs}
        unassigned_ip_addresses_names: set = set(all_ip_addresses.keys()).difference(assigned_ip_addresses)
        
        ## if there are more unassigned ip addresses than number of reserved ip addresses
        ## remove some unassigned ip addresses
        while len(unassigned_ip_addresses_names) > self.num_reserved_proxies:
            ip_name: str = unassigned_ip_addresses_names.pop()
            self._remove_public_ip_address(ip_name, wait=False)
        
        unassigned_ip_addresses_list: list = [all_ip_addresses.get(ip_addr_name) for ip_addr_name in unassigned_ip_addresses_names]
        return unassigned_ip_addresses_list
   
   
    def _get_vms_ip_addresses(self) -> list[str]:
        self._get_state()
        vms: list[VirtualMachine] = list(deepcopy((self.__virtual_machines)))
        ip_addrs: list[str] = []
        for vm in vms:
            nics: list[NetworkInterface] = list(deepcopy(vm.network_profile.network_interfaces))
            for nic_ref in nics:
                pprint(nic_ref.__dict__)
                # Get the network interface
                nic = self.__network_client.network_interfaces.get(
                    resource_group_name=azure_config.RESOURCE_GROUP, 
                    network_interface_name=nic_ref.id.rsplit('/',1)[-1],
                )
                for ipconfig in nic.ip_configurations:
                    public_ip_address = ipconfig.public_ip_address
                    if public_ip_address is not None:
                        print(public_ip_address.__dict__)
                    ip_addrs.append(public_ip_address)
                    
        return ip_addrs
    
    
    """Debug methods"""
    
    def _print_counters(self) -> None:
        self._get_state()
        print('__virtual_machines:',len(list(deepcopy(self.__virtual_machines))))
        print('__network_interfaces:',len(list(deepcopy(self.__network_interfaces))))
        print('__ip_configs:',len(list(deepcopy(self.__ip_configs))))
        print('__subnets:',len(list(deepcopy(self.__subnets))))
        print('__ip_addrs:',len(list(deepcopy(self.__ip_addrs))))
        print()
        
            
"""Testing"""
if __name__ == '__main__':
    proxy: Proxy = AzureProxy(
        num_active_proxies=3,
        num_reserved_proxies=6,
        start_vm=False,
    )
    
    proxy._print_counters()
    
    # proxy._remove_public_ip_address()
    
    exit()
    active_ips = proxy.get_proxies()
    ic(proxy.get_proxies())
    
    proxy.replace_proxy(active_ips[0])
    ic(proxy.get_proxies())
    
    active_ips = proxy.get_proxies()
    proxy.replace_proxy(active_ips[0])
    ic(proxy.get_proxies())
    
    active_ips = proxy.get_proxies()
    proxy.replace_proxy(active_ips[0])
    ic(proxy.get_proxies())
    
    active_ips = proxy.get_proxies()
    proxy.replace_proxy(active_ips[0])
    ic(proxy.get_proxies())
    
    # old_public_ip_addr = '172.190.239.71'
    # new_public_ip_addr = proxy.replace_ip_address(old_public_ip_addr)
    # ic(old_public_ip_addr,new_public_ip_addr)
    # proxy._print_counters()
    
    # proxy._create_network_interface()
    
    # pprint(proxy._get_ip_from_vm(vm_name='vm01'))
    # proxy._remove_public_ip_address('vm01-ip')
    # proxy._print_counters()
    
    # pprint(proxy._associate_public_ip_to_nic(nic='nic_roxy',public_ip='ip_amy'))
    
    # proxy._display_iterables(proxy._AzureProxy__ip_configs)
    # proxy._display_iterables(proxy._AzureProxy__ip_addrs)
    # proxy._display_iterables(proxy._AzureProxy__network_interfaces)
    # proxy._display_iterables(proxy._AzureProxy__virtual_machines)
    # pprint(proxy._get_vms_ip_addresses())
    
    # proxy.stop()