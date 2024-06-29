import redis
import json
import threading
import time
from cryptography.fernet import Fernet
from base64 import urlsafe_b64encode, urlsafe_b64decode


def generate_key():
    # generate by chatgpt-4
    # generate Fernet key
    return Fernet.generate_key()


def encrypt_str_data(str_data, key):
    # generate by chatgpt-4
    # encrypt string data
    f = Fernet(key)
    str_bytes = str_data.encode("utf-8")
    encrypted_str = f.encrypt(str_bytes)
    return encrypted_str


def decrypt_str_data(encrypted_str, key):
    # generate by chatgpt-4
    # decrypt string data
    f = Fernet(key)
    decrypted_str_bytes = f.decrypt(encrypted_str)
    decrypted_str = decrypted_str_bytes.decode("utf-8")
    return decrypted_str


def encrypt_json_data(json_data, key):
    # generate by chatgpt-4
    # encrypt json data,also string
    f = Fernet(key)
    json_str = json.dumps(json_data)
    json_bytes = json_str.encode("utf-8")
    encrypted_json = f.encrypt(json_bytes)
    return encrypted_json


def decrypt_json_data(encrypted_json, key):
    # generate by chatgpt-4
    # decrypt data
    f = Fernet(key)
    # encrypted_json must be bytes not str!!!
    decrypted_json_bytes = f.decrypt(encrypted_json)
    decrypted_json_str = decrypted_json_bytes.decode("utf-8")
    decrypted_json_data = json.loads(decrypted_json_str)
    return decrypted_json_data


def convert_dict_byte2str(byte_dict, value_dict=True, key_encrypt="", value_encrypt=""):
    converted_data = {}
    for key, value in byte_dict.items():
        # Convert keys and values from bytes to str.
        if key_encrypt == "":
            str_key = key.decode('utf-8')
        else:
            str_key = decrypt_str_data(key, key_encrypt)

        if value_encrypt == "":
            if value_dict:
                str_value = json.loads(value.decode('utf-8'))
            else:
                str_value = value.decode('utf-8')
        else:
            if value_dict:
                str_value = decrypt_json_data(value, value_encrypt)
            else:
                str_value = decrypt_str_data(value, value_encrypt)
        converted_data[str_key] = str_value
    return converted_data


class RedisApi():
    '''
    {
        "host": "127.0.0.1",
        "port": 7777,
        "user": "test_rw",
        "psw": "test123",
        "db": 0,
        "channel": "test_chn"
    }
    '''

    def __init__(self, config: dict) -> None:
        # decode_responses=self.encrypt the Fernet en de should input bytes!
        self.config = config
        self.s = self.create_redis_client()

    def create_redis_client(self):
        return redis.Redis(host=self.config["host"],
                           port=self.config["port"],
                           username=self.config['user'],
                           password=self.config['password'],
                           db=self.config['db'],
                           socket_keepalive=True,
                           socket_timeout=5)

    def select_db(self, db):
        # When switching databases, reconnect to the new database.
        self.config['db'] = db
        self.s = self.create_redis_client()

    def check_connection(self):
        try:
            return self.s.ping()
        except redis.ConnectionError:
            return False

    def publish_data(self, channel, data):
        ret = self.s.publish(channel, data)
        if ret > 0:
            print("publish {} client success!".format(ret))
        else:
            print("publish fail!")

    def subscribe(self, channel, callback, sleep=0.001):
        # todo:need to handle disconnection and re-subscription
        pubsub = self.s.pubsub()
        pubsub.subscribe(
            **{channel: lambda message: callback(message["data"])})

        def run_pubsub(pubsub_instance):
            pubsub_instance.run_in_thread(sleep_time=sleep)

        pubsub_thread = threading.Thread(target=run_pubsub, args=(pubsub,))
        pubsub_thread.start()

    def subscribe(self, channel, callback, stop_event, sleep=0.001, timeout=600):
        def run_pubsub():
            while not stop_event.is_set():
                try:

                    last_message_time = time.time()
                    pubsub = self.s.pubsub()
                    pubsub.psubscribe(channel)

                    while not stop_event.is_set():
                        if self.check_connection() == False:
                            print("{}:run_pubsub:connection lost,try reconnect!".format(
                                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
                            time.sleep(1)
                            break
                        try:
                            message = pubsub.get_message()
                            if message:
                                if message["type"] == "message" or message["type"] == "pmessage":
                                    callback(message['data'])
                                else:
                                    print("channel {} recv data:{}".format(
                                        channel, message))
                                last_message_time = time.time()
                            elif time.time() - last_message_time > timeout:
                                print("{}:run_pubsub:no message timeout,try reconnect!".format(
                                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
                                break
                            time.sleep(sleep)
                        except Exception as e:
                            print(
                                "pubsub.get_message exception!msg:{}".format(e))
                            break

                    time.sleep(sleep)
                    # do not recreate a new redis instance
                    # self.s = self.create_redis_client()  # Reconnect

                except Exception as e:
                    print("run_pubsub exception!msg:{}".format(e))
                    time.sleep(1)
                    # do not recreate a new redis instance
                    # self.s = self.create_redis_client()  # Reconnect

        pubsub_thread = threading.Thread(target=run_pubsub)
        pubsub_thread.daemon = True
        pubsub_thread.start()

    def subscribe_decorator(self, channel):
        def decorator(f):
            def wrapper(*args, **kwargs):
                self.subscribe(channel, f)
            return wrapper
        return decorator

    def update_hashtable(self, hash, key, value):
        ret = self.s.hset(hash, key, value)
        if ret != 1 and ret != 0:
            print("update hashtable error!ret={},hash={},key={},value={}".format(
                ret, hash, key, value))

    def get_all_keys(self):
        return self.s.keys("*")

    def get_all_keys_and_show(self):
        for key in self.get_all_keys():
            rtype = self.s.type(key).decode('utf-8')
            print(rtype)
            if rtype == 'string':
                value = self.s.get(key)
            elif rtype == 'hash':
                value = self.s.hgetall(key)
            elif rtype == 'list':
                value = self.s.lrange(key, 0, -1)
            elif rtype == 'set':
                value = self.s.smembers(key)
            elif rtype == 'zset':
                value = self.s.zrange(key, 0, -1, withscores=True)
            else:
                value = 'unkown type'

            print(f"{key}: {value}")

    def get_from_hashtable(self, hash, *keys):
        if not keys:
            return self.s.hgetall(hash)

        raw_values = self.s.hmget(hash, keys)
        result = {}
        for key, value in zip(keys, raw_values):
            if value is not None:
                result[key] = value

        return result

    def get_hash_keys(self, hash):
        return self.s.hkeys(hash)

    def delete_hashtable(self, hash):
        ret = self.s.delete(hash)
        if ret != 1:
            print("delete hash fail!ret={},hash={}".format(ret, hash))

    def delete_from_hashtable(self, hash, *keys):
        if len(keys) == 1:
            ret = self.s.hdel(hash, keys[0])
        elif len(keys) > 1:
            ret = self.s.hdel(hash, *keys)
        else:
            ret = self.s.delete(hash)

        if ret == 0:
            print("nothing delete.hash={},keys={}".format(hash, keys))

    def clear_all_data(self):
        self.s.flushall()

    def clear_db_data(self):
        self.s.flushdb()

    def polling_hashtable(self, hashtable, callback, stop_event, sleep=1):
        def run_polling(api, hashtable):
            while not stop_event.is_set():
                try:
                    message = api.get_from_hashtable(hashtable)
                    if message:
                        callback(message, True)
                except Exception as e:
                    msg = "redis get hashtable fail!({})".format(e)
                    callback({"msg": msg, "ts": time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime())}, False)
                time.sleep(sleep)

        polling_thread = threading.Thread(
            target=run_polling, args=(self, hashtable,))
        polling_thread.daemon = True
        polling_thread.start()


if __name__ == "__main__":
    print(generate_key())
