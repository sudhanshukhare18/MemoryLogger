import secrets
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.asymmetric import rsa,padding
from cryptography.hazmat.primitives import hashes

#symmetric encryption

def aes_ed(message):
    key = secrets.token_bytes(32)
    nonce = secrets.token_bytes(12)
    aes = AESGCM(key)

    cipertext = nonce + aes.encrypt(nonce,message.encode(),None)
    plaintext = aes.decrypt(ciphertext[:12],ciphertext[12:],None)
    return key.hex(), ciphertext.hex(), plaintext.decode()


#Asymmetric ecryption

def rsa_ed(message):
    private_key = rsa.generate_private_key(public_exponent=65537 , key_size=2048)
    public_key = private_key.public_key()
    ciphertext = public_key.encrypt(
        message.encode(),
        def rsa_ed
plaintext = private_key.decrypt(
    cipertext,
    padding.OAEP(
        mgf=padding.MGF1(algorithm=hashes.SHA256()),
        algorithm=hashes.SHA256(),
        label=None
    )
)
def enc(a){
    return aes_ed(a)
}

if __name__== "__main__":
    print(aes_ed("tushar  "))