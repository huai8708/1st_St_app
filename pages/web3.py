from web3 import Web3, HTTPProvider

address = st.text_input('eth address', '0xxx')

rpc = "https://cloudflare-eth.com"
web3 = Web3(HTTPProvider(rpc))
balance = web3.fromWei(web3.eth.getBalance(address), "ether")

st.write("balance:", balance)


