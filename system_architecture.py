import graphviz

def create_system_architecture():
    """Create a system architecture diagram showing the complete fall detection pipeline."""
    dot = graphviz.Digraph(comment='Fall Detection System Architecture')
    dot.attr(rankdir='TB')
    
    # Add nodes with styling
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
    
    # Input Layer
    with dot.subgraph(name='cluster_0') as c:
        c.attr(label='Input Layer')
        c.node('video_input', 'Video Input')
        c.node('audio_input', 'Audio Input')
    
    # Processing Layer
    with dot.subgraph(name='cluster_1') as c:
        c.attr(label='Processing Layer')
        c.node('mediapipe', 'MediaPipe Pose\nEstimation')
        c.node('panns', 'PANNs Audio\nProcessing')
        c.node('feature_extraction', 'Feature Extraction\n(Base, Temporal, Domain)')
    
    # Model Layer
    with dot.subgraph(name='cluster_2') as c:
        c.attr(label='Model Layer')
        c.node('video_model', 'VideoModel\n(CNN + Attention)')
        c.node('audio_model', 'AudioModel\n(WavLM + Transformer)')
        c.node('fusion', 'CrossModal\nAttention Fusion')
        c.node('ensemble', 'Ensemble Model\n(PANNs + LGBM + RF)')
    
    # Output Layer
    with dot.subgraph(name='cluster_3') as c:
        c.attr(label='Output Layer')
        c.node('state_machine', 'State Machine\n(Normal/Detected/Fallen)')
        c.node('output', 'Fall Detection\nOutput')
    
    # Add edges
    dot.edge('video_input', 'mediapipe')
    dot.edge('audio_input', 'panns')
    dot.edge('mediapipe', 'feature_extraction')
    dot.edge('panns', 'feature_extraction')
    dot.edge('feature_extraction', 'video_model')
    dot.edge('feature_extraction', 'audio_model')
    dot.edge('video_model', 'fusion')
    dot.edge('audio_model', 'fusion')
    dot.edge('fusion', 'ensemble')
    dot.edge('ensemble', 'state_machine')
    dot.edge('state_machine', 'output')
    
    # Save the diagram
    dot.render('system_architecture', format='png', cleanup=True)
    print("System architecture diagram saved as 'system_architecture.png'")

def create_model_architecture():
    """Create detailed model architecture diagrams for each component."""
    # Video Model Architecture
    video_dot = graphviz.Digraph(comment='Video Model Architecture')
    video_dot.attr(rankdir='TB')
    
    with video_dot.subgraph(name='cluster_0') as c:
        c.attr(label='Video Model')
        c.node('input', 'Input\n(33 keypoints)')
        c.node('cnn', 'CNN Layers')
        c.node('attention', 'Self-Attention')
        c.node('output', 'Output\n(Fall Probability)')
    
    video_dot.edge('input', 'cnn')
    video_dot.edge('cnn', 'attention')
    video_dot.edge('attention', 'output')
    video_dot.render('video_model_architecture', format='png', cleanup=True)
    
    # Audio Model Architecture
    audio_dot = graphviz.Digraph(comment='Audio Model Architecture')
    audio_dot.attr(rankdir='TB')
    
    with audio_dot.subgraph(name='cluster_0') as c:
        c.attr(label='Audio Model')
        c.node('input', 'Input\n(Audio Waveform)')
        c.node('wavlm', 'WavLM Base')
        c.node('transformer', 'Transformer\nLayers')
        c.node('output', 'Output\n(Fall Probability)')
    
    audio_dot.edge('input', 'wavlm')
    audio_dot.edge('wavlm', 'transformer')
    audio_dot.edge('transformer', 'output')
    audio_dot.render('audio_model_architecture', format='png', cleanup=True)
    
    print("Model architecture diagrams saved as 'video_model_architecture.png' and 'audio_model_architecture.png'")

if __name__ == "__main__":
    create_system_architecture()
    create_model_architecture() 