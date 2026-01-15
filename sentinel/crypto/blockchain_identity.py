# Blockchain Identity Module
# Implements decentralized identity using blockchain

import json
import os
import hashlib
import time
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class BlockchainIdentity:
    """
    Implements blockchain-based decentralized identity (DID).
    Features:
    - Self-sovereign identity
    - Verifiable credentials
    - Decentralized identifiers (DIDs)
    - Immutable identity records
    """
    
    def __init__(self, storage_dir="data/models"):
        self.storage_dir = storage_dir
        self.blockchain_file = os.path.join(storage_dir, "identity_blockchain.json")
        self.did_registry_file = os.path.join(storage_dir, "did_registry.json")
        
        # Blockchain
        self.chain = []
        
        # DID Registry
        self.did_registry = {}
        
        # Verifiable credentials
        self.credentials = {}
        
        # Load existing data
        self.load_blockchain()
        self.load_did_registry()
        
        # Create genesis block if needed
        if len(self.chain) == 0:
            self.create_genesis_block()
    
    def load_blockchain(self):
        """Load blockchain from disk"""
        if os.path.exists(self.blockchain_file):
            try:
                with open(self.blockchain_file, 'r') as f:
                    self.chain = json.load(f)
            except Exception as e:
                logger.error(f"Error loading blockchain: {e}")
                self.chain = []
    
    def save_blockchain(self):
        """Save blockchain to disk"""
        try:
            os.makedirs(self.storage_dir, exist_ok=True)
            with open(self.blockchain_file, 'w') as f:
                json.dump(self.chain, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving blockchain: {e}")
    
    def load_did_registry(self):
        """Load DID registry"""
        if os.path.exists(self.did_registry_file):
            try:
                with open(self.did_registry_file, 'r') as f:
                    data = json.load(f)
                    self.did_registry = data.get('registry', {})
                    self.credentials = data.get('credentials', {})
            except Exception as e:
                logger.error(f"Error loading DID registry: {e}")
    
    def save_did_registry(self):
        """Save DID registry"""
        try:
            with open(self.did_registry_file, 'w') as f:
                json.dump({
                    'registry': self.did_registry,
                    'credentials': self.credentials
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving DID registry: {e}")
    
    def create_genesis_block(self):
        """Create the genesis block"""
        genesis_block = {
            'index': 0,
            'timestamp': time.time(),
            'transactions': [],
            'previous_hash': '0',
            'nonce': 0
        }
        
        genesis_block['hash'] = self._calculate_hash(genesis_block)
        self.chain.append(genesis_block)
        self.save_blockchain()
        
        logger.info("Genesis block created")
    
    def create_did(self, user_id: str, public_key: str, metadata: dict = None) -> str:
        """
        Create a Decentralized Identifier (DID)
        
        Args:
            user_id: User identifier
            public_key: User's public key
            metadata: Additional metadata
        
        Returns:
            DID string
        """
        # Generate DID
        did = f"did:brocode:{hashlib.sha256(user_id.encode()).hexdigest()[:32]}"
        
        # Create DID document
        did_document = {
            'id': did,
            'created': time.time(),
            'updated': time.time(),
            'publicKey': [{
                'id': f"{did}#keys-1",
                'type': 'Ed25519VerificationKey2018',
                'controller': did,
                'publicKeyHex': public_key
            }],
            'authentication': [f"{did}#keys-1"],
            'service': [],
            'metadata': metadata or {}
        }
        
        # Store in registry
        self.did_registry[did] = did_document
        self.save_did_registry()
        
        # Add to blockchain
        transaction = {
            'type': 'DID_CREATION',
            'did': did,
            'timestamp': time.time(),
            'user_id': user_id
        }
        
        self.add_block([transaction])
        
        logger.info(f"Created DID: {did}")
        
        return did
    
    def resolve_did(self, did: str) -> Optional[dict]:
        """
        Resolve a DID to its document
        
        Args:
            did: Decentralized identifier
        
        Returns:
            DID document or None
        """
        return self.did_registry.get(did)
    
    def issue_credential(self, issuer_did: str, subject_did: str, credential_type: str, claims: dict) -> str:
        """
        Issue a verifiable credential
        
        Args:
            issuer_did: DID of credential issuer
            subject_did: DID of credential subject
            credential_type: Type of credential
            claims: Claims in the credential
        
        Returns:
            Credential ID
        """
        # Generate credential ID
        credential_id = hashlib.sha256(
            f"{issuer_did}{subject_did}{time.time()}".encode()
        ).hexdigest()
        
        # Create verifiable credential
        credential = {
            '@context': ['https://www.w3.org/2018/credentials/v1'],
            'id': f"urn:uuid:{credential_id}",
            'type': ['VerifiableCredential', credential_type],
            'issuer': issuer_did,
            'issuanceDate': time.time(),
            'credentialSubject': {
                'id': subject_did,
                **claims
            },
            'proof': {
                'type': 'Ed25519Signature2018',
                'created': time.time(),
                'proofPurpose': 'assertionMethod',
                'verificationMethod': f"{issuer_did}#keys-1",
                'jws': self._create_proof(issuer_did, claims)
            }
        }
        
        # Store credential
        if subject_did not in self.credentials:
            self.credentials[subject_did] = []
        
        self.credentials[subject_did].append(credential)
        self.save_did_registry()
        
        # Add to blockchain
        transaction = {
            'type': 'CREDENTIAL_ISSUED',
            'credential_id': credential_id,
            'issuer': issuer_did,
            'subject': subject_did,
            'timestamp': time.time()
        }
        
        self.add_block([transaction])
        
        logger.info(f"Issued credential: {credential_id}")
        
        return credential_id
    
    def verify_credential(self, credential_id: str) -> bool:
        """
        Verify a credential
        
        Args:
            credential_id: Credential ID to verify
        
        Returns:
            Boolean indicating if credential is valid
        """
        # Find credential
        for did, creds in self.credentials.items():
            for cred in creds:
                if cred['id'] == f"urn:uuid:{credential_id}":
                    # Verify proof
                    issuer_did = cred['issuer']
                    issuer_doc = self.resolve_did(issuer_did)
                    
                    if not issuer_doc:
                        return False
                    
                    # Verify signature (simplified)
                    # In production, use actual signature verification
                    return True
        
        return False
    
    def get_user_credentials(self, did: str) -> List[dict]:
        """Get all credentials for a DID"""
        return self.credentials.get(did, [])
    
    def revoke_credential(self, credential_id: str, reason: str = 'unspecified'):
        """
        Revoke a credential
        
        Args:
            credential_id: Credential ID to revoke
            reason: Reason for revocation
        """
        # Add revocation to blockchain
        transaction = {
            'type': 'CREDENTIAL_REVOKED',
            'credential_id': credential_id,
            'reason': reason,
            'timestamp': time.time()
        }
        
        self.add_block([transaction])
        
        logger.info(f"Revoked credential: {credential_id}")
    
    def add_block(self, transactions: List[dict]):
        """
        Add a new block to the blockchain
        
        Args:
            transactions: List of transactions
        """
        previous_block = self.chain[-1]
        
        new_block = {
            'index': len(self.chain),
            'timestamp': time.time(),
            'transactions': transactions,
            'previous_hash': previous_block['hash'],
            'nonce': 0
        }
        
        # Proof of work (simplified)
        new_block['nonce'] = self._proof_of_work(new_block)
        new_block['hash'] = self._calculate_hash(new_block)
        
        self.chain.append(new_block)
        self.save_blockchain()
        
        logger.info(f"Added block {new_block['index']} to blockchain")
    
    def verify_chain(self) -> bool:
        """Verify the integrity of the blockchain"""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            
            # Verify hash
            if current_block['hash'] != self._calculate_hash(current_block):
                return False
            
            # Verify chain
            if current_block['previous_hash'] != previous_block['hash']:
                return False
        
        return True
    
    def _calculate_hash(self, block: dict) -> str:
        """Calculate hash of a block"""
        block_copy = block.copy()
        block_copy.pop('hash', None)
        
        block_string = json.dumps(block_copy, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def _proof_of_work(self, block: dict, difficulty: int = 2) -> int:
        """
        Simple proof of work
        
        Args:
            block: Block to mine
            difficulty: Number of leading zeros required
        
        Returns:
            Nonce that satisfies proof of work
        """
        nonce = 0
        target = '0' * difficulty
        
        while True:
            block['nonce'] = nonce
            hash_result = self._calculate_hash(block)
            
            if hash_result[:difficulty] == target:
                return nonce
            
            nonce += 1
            
            # Limit iterations for demo
            if nonce > 10000:
                return nonce
    
    def _create_proof(self, issuer_did: str, claims: dict) -> str:
        """Create cryptographic proof for credential"""
        # Simplified proof creation
        # In production, use actual digital signatures
        proof_data = f"{issuer_did}{json.dumps(claims, sort_keys=True)}"
        return hashlib.sha256(proof_data.encode()).hexdigest()
    
    def get_statistics(self) -> dict:
        """Get blockchain identity statistics"""
        total_dids = len(self.did_registry)
        total_credentials = sum(len(creds) for creds in self.credentials.values())
        chain_length = len(self.chain)
        chain_valid = self.verify_chain()
        
        # Count transaction types
        transaction_types = {}
        for block in self.chain[1:]:  # Skip genesis
            for tx in block['transactions']:
                tx_type = tx.get('type', 'UNKNOWN')
                transaction_types[tx_type] = transaction_types.get(tx_type, 0) + 1
        
        return {
            'total_dids': total_dids,
            'total_credentials': total_credentials,
            'blockchain_length': chain_length,
            'blockchain_valid': chain_valid,
            'transaction_types': transaction_types,
            'did_method': 'did:brocode'
        }
