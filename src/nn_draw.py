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
      <canvas id=\"canvas\" style=\"display:block; margin:0 auto; max-width:1000px; width:1000px; height:700px;\" width=\"1000\" height=\"700\"></canvas>
      <script>
        const canvas = document.getElementById("canvas");
        const ctx = canvas.getContext("2d");
        const layers = {layers};
        const values = {values};
        const outputProbs = {probs};
        const layerX = [100, 350, 600, 850];
        const radius = 22;
        const outputLayer = values[values.length - 1];
        let maxIdx = 0;
        for (let i = 1; i < outputLayer.length; i++) {{
          if (outputLayer[i] > outputLayer[maxIdx]) maxIdx = i;
        }}
        function getNeuronY(layerIndex, neuronIndex) {{
          const count = layers[layerIndex];
          const spacing = 620 / (count - 1);
          const offset = 40;
          return offset + neuronIndex * spacing;
        }}
        function draw() {{
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          // Draw all connections
          for (let l = 0; l < layers.length - 1; l++) {{
            for (let i = 0; i < layers[l]; i++) {{
              for (let j = 0; j < layers[l+1]; j++) {{
                const x1 = layerX[l];
                const y1 = getNeuronY(l, i);
                const x2 = layerX[l+1];
                const y2 = getNeuronY(l+1, j);
                ctx.beginPath();
                ctx.moveTo(x1, y1);
                ctx.lineTo(x2, y2);
                ctx.strokeStyle = '#2ecc40';
                ctx.lineWidth = 2;
                ctx.stroke();
              }}
            }}
          }}
          // Draw neurons
          for (let l = 0; l < layers.length; l++) {{
            for (let i = 0; i < layers[l]; i++) {{
              const x = layerX[l];
              const y = getNeuronY(l, i);
              ctx.save();
              ctx.beginPath();
              ctx.arc(x, y, radius, 0, 2 * Math.PI);
              let fill = '#111';
              // Only activate output neuron if probability >= 0.1%
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
              // Output layer: draw probability and label
              if (l === layers.length - 1) {{
                ctx.save();
                // Probability as percentage inside neuron
                ctx.fillStyle = '#fff';
                ctx.font = 'bold 13px monospace';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText((outputProbs[i]*100).toFixed(1) + '%', x, y);
                // Label ao lado direito (maior e mais espaçado)
                ctx.fillStyle = '#fff';
                ctx.font = 'bold 28px monospace';
                ctx.textAlign = 'left';
                ctx.textBaseline = 'middle';
                ctx.fillText(i.toString(), x + radius + 28, y);
                if (i === maxIdx) {{
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
          canvas.width = 1000;
          canvas.height = 700;
          draw();
        }}
        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();
      </script>
    </div>
    """, height=750)
