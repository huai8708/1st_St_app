import streamlit.components.v1 as components  # Import Streamlit
import streamlit as st

dd = '''
<html><head>
<script src="https://aframe.io/releases/1.0.4/aframe.min.js"></script>
<script src="https://unpkg.com/aframe-environment-component@1.1.0/dist/aframe-environment-component.min.js"></script>
<script src="https://unpkg.com/aframe-event-set-component@4.2.1/dist/aframe-event-set-component.min.js"></script>
</head><body>
<a-scene>
<a-box position="-1 0.5 -3" rotation="0 0 0" color="#4CC3D9"></a-box>

<a-light type=ambient color="red" position="0 5 0"></a-light>

<a-entity environment="preset: forest; groundColor: #445; grid: cross"></a-entity>

<a-scene fog="type: exponential; color: #AAA"></a-scene>

</a-scene>
</body></html>
'''


components.html(dd)