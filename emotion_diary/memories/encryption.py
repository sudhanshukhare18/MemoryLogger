# memories/encryption.py
import os
import base64
from typing import Tuple, Optional
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# Load AES key from env (base64 encoded). Fallback for dev only (32 bytes).

    # DEV fallback (32 bytes) — replace in production!
AES_KEY = b"0123456789ABCDEF0123456789ABCDEF"

if len(AES_KEY) not in (16, 24, 32):
    raise ValueError(f"AES key must be 16/24/32 bytes long. Got {len(AES_KEY)} bytes.")

def encrypt_aes(plaintext: str) -> str:
    """
    Encrypts plaintext (UTF-8) using AES-GCM.
    Returns base64 string that encodes nonce + ciphertext + tag:
      b64 = base64.b64encode(nonce || ciphertext).decode('ascii')
    Nonce is 12 bytes (recommended).
    """
    if not isinstance(plaintext, str):
        raise TypeError("plaintext must be str")
    aesgcm = AESGCM(AES_KEY)
    nonce = os.urandom(12)  # AESGCM recommended 12-byte nonce
    ct = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), associated_data=None)
    combined = nonce + ct
    return base64.b64encode(combined).decode("ascii")

def decrypt_aes(b64_string: str) -> str:
    """
    Decrypts the string produced by encrypt_aes.
    Returns plaintext str.
    Raises ValueError on bad input or decryption failure.
    """
    if not isinstance(b64_string, str):
        raise TypeError("b64_string must be str")
    try:
        combined = base64.b64decode(b64_string)
    except Exception as e:
        raise ValueError(f"base64 decode failed: {e}")

    if len(combined) < 13:
        # nonce (12) + at least 1 byte ciphertext
        raise ValueError("ciphertext too short or corrupted")

    nonce = combined[:12]
    ct = combined[12:]
    aesgcm = AESGCM(AES_KEY)
    try:
        pt = aesgcm.decrypt(nonce, ct, associated_data=None)
        return pt.decode("utf-8")
    except Exception as e:
        raise ValueError(f"AES decryption failed: {e}")

# Optional helpers: try decoding hex too (for legacy)
def try_decrypt_various_formats(raw: str) -> str:
    """
    Try common encodings: base64 (primary), hex fallback.
    Returns plaintext or raises ValueError.
    """
    # 1) try base64
    try:
        return decrypt_aes(raw)
    except Exception as e_b64:
        # try hex
        try:
            raw_bytes = bytes.fromhex(raw)
            combined_b64 = base64.b64encode(raw_bytes).decode("ascii")
            return decrypt_aes(combined_b64)
        except Exception as e_hex:
            raise ValueError(f"Not decryptable: base64_err={e_b64}; hex_err={e_hex}")
