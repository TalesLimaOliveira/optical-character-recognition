import streamlit as st
import torch

def draw_network_visualization(model, img_tensor):
    with torch.no_grad():
        output, h1, h2 = model(img_tensor.view(1, 1, 28, 28))

    # Extract a subset of activations for visualization
    input_flat = img_tensor.view(-1)[:18].tolist()
    h1_sub = h1.view(-1)[:14].tolist()
    h2_sub = h2.view(-1)[:14].tolist()
    out_sub = output.view(-1).tolist()

    # Only show weights for neuron 1 (index 0) of each layer
    weights1 = [[w[0] for w in model.fc1.weight[:14, :18].tolist()]]
    weights2 = [[w[0] for w in model.fc2.weight[:14, :14].tolist()]]
    weights3 = [[w[0] for w in model.fc3.weight[:10, :14].tolist()]]

    layers = [18, 14, 14, 10]
    values = [input_flat, h1_sub, h2_sub, out_sub]
    weights = [weights1, weights2, weights3]
    activated_idx = out_sub.index(max(out_sub))

    # Calculate probabilities for output neurons
    import numpy as np
    probs = np.exp(out_sub) / np.sum(np.exp(out_sub))
    probs = probs.tolist()

    st.markdown("""
    <div style='position: relative; width: 1000px; height: 40px; margin: 0 auto;'>
        <div style='position: absolute; left: 75px; top: 0; width: 0; text-align: center; font-weight: bold; font-size: 1.2em; color: #fff;'>INPUT</div>
        <div style='position: absolute; left: 375px; top: 0; width: 200px; text-align: center; font-weight: bold; font-size: 1.2em; color: #fff; white-space: nowrap;'>HIDDEN LAYERS</div>
        <div style='position: absolute; left: 812px; top: 0; width: 0; text-align: center; font-weight: bold; font-size: 1.2em; color: #fff;'>OUTPUT</div>
    </div>
    <div style='position: relative; width: 1000px; height: 30px; margin: 0 auto;'>
        <div style='position: absolute; left: 40px; top: 0; width: 200; text-align: center; font-size: 1.1em; color: #bbb;'>784 Image Pixels</div>
        <div style='position: absolute; left: 315px; top: 0; width: 200; text-align: center; font-size: 1.1em; color: #bbb;'>128 ReLU</div>
        <div style='position: absolute; left: 570px; top: 0; width: 200; text-align: center; font-size: 1.1em; color: #bbb;'>64 ReLU</div>
        <div style='position: absolute; left: 807px; top: 0; width: 200; text-align: center; font-size: 1.1em; color: #bbb;'>10 Softmax</div>
    </div>
    """, unsafe_allow_html=True)

    st.components.v1.html(f"""
    <div style='display: flex; justify-content: center; width: 100%;'>
      <div id="canvas-container" style="width:100%; max-width:1000px; min-width:320px;">
        <canvas id="canvas" style="display:block; margin:0 auto; width:100%; height:auto; background:transparent;" width="1000" height="700"></canvas>
      </div>
      <script>
        function waitForCanvasAndDraw() {{
          var canvas = document.getElementById('canvas');
          if (!canvas) {{
            setTimeout(waitForCanvasAndDraw, 100);
            return;
          }}
          var ctx = canvas.getContext('2d');
          var layers = {layers};
          var values = {values};
          var outputProbs = {probs};
          function getCanvasWidth() {{
            return canvas.parentElement.offsetWidth;
          }}
          function getCanvasHeight() {{
            return Math.round(getCanvasWidth() * 0.7);
          }}
          function getLayerX(width) {{
            return [0.1, 0.35, 0.6, 0.85].map(function(f) {{ return Math.round(width * f); }});
          }}
          function getNeuronY(layerIndex, neuronIndex, height) {{
            var count = layers[layerIndex];
            var spacing = (height - 80) / (count - 1);
            var offset = 40;
            return offset + neuronIndex * spacing;
          }}
          function draw() {{
            var width = getCanvasWidth();
            var height = getCanvasHeight();
            canvas.width = width;
            canvas.height = height;
            ctx.clearRect(0, 0, width, height);
            var layerX = getLayerX(width);
            var radius = Math.max(18, Math.min(28, Math.round(width/50)));
            for (var l = 0; l < layers.length - 1; l++) {{
              for (var i = 0; i < layers[l]; i++) {{
                for (var j = 0; j < layers[l+1]; j++) {{
                  var x1 = layerX[l];
                  var y1 = getNeuronY(l, i, height);
                  var x2 = layerX[l+1];
                  var y2 = getNeuronY(l+1, j, height);
                  ctx.beginPath();
                  ctx.moveTo(x1, y1);
                  ctx.lineTo(x2, y2);
                  ctx.strokeStyle = '#2ecc40';
                  ctx.lineWidth = 2;
                  ctx.stroke();
                }}
              }}
            }}
            for (var l = 0; l < layers.length; l++) {{
              for (var i = 0; i < layers[l]; i++) {{
                var x = layerX[l];
                var y = getNeuronY(l, i, height);
                ctx.save();
                ctx.beginPath();
                ctx.arc(x, y, radius, 0, 2 * Math.PI);
                var fill = '#111';
                if (l === layers.length - 1) {{
                  if (outputProbs[i] >= 0.001) {{
                    fill = '#2ecc40';
                  }}
                }} else {{
                  if (values[l][i] > 0) {{
                    fill = '#2ecc40';
                  }}
                }}
                ctx.fillStyle = fill;
                ctx.fill();
                ctx.strokeStyle = '#2ecc40';
                ctx.lineWidth = 3;
                ctx.stroke();
                ctx.restore();
                if (l === layers.length - 1) {{
                  ctx.save();
                  ctx.fillStyle = '#fff';
                  ctx.font = 'bold ' + Math.max(11, Math.round(width/80)) + 'px monospace';
                  ctx.textAlign = 'center';
                  ctx.textBaseline = 'middle';
                  ctx.fillText((outputProbs[i]*100).toFixed(1) + '%', x, y);
                  ctx.fillStyle = '#fff';
                  ctx.font = 'bold ' + Math.max(22, Math.round(width/35)) + 'px monospace';
                  ctx.textAlign = 'left';
                  ctx.textBaseline = 'middle';
                  ctx.fillText(i.toString(), x + radius + Math.max(18, Math.round(width/55)), y);
                  if (i === outputProbs.indexOf(Math.max.apply(null, outputProbs))) {{
                    ctx.beginPath();
                    ctx.arc(x, y, radius + 3, 0, 2 * Math.PI);
                    ctx.strokeStyle = '#0099ff';
                    ctx.lineWidth = 5;
                    ctx.stroke();
                  }}
                  ctx.restore();
                }}
              }}
            }}
          }}
          function resizeCanvas() {{
            draw();
          }}
          window.addEventListener('resize', resizeCanvas);
          setTimeout(resizeCanvas, 100);
          draw();
        }}
        waitForCanvasAndDraw();
      </script>
    </div>
    """, height=750)
