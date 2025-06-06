import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import PyPDF2
import docx
import tempfile
import os
import sys
import importlib.util

# Import the model from main_model.py
def import_model_from_file(file_path):
    module_name = os.path.basename(file_path).replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Assume main_model.py is in the same directory
main_model = import_model_from_file("main_model_copy.py")
EnhancedQuestionProbabilityModel = main_model.EnhancedQuestionProbabilityModel
prepare_enhanced_dataset = main_model.prepare_enhanced_dataset

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# Function to extract questions from text
def extract_questions(text):
    # Different patterns to identify questions
    # This is a simple approach; might need refinement based on document format
    patterns = [
        r'\d+\.\s+(.*?)\n',  # numbered questions like "1. What is..."
        r'[A-Z][^.!?]*\?',   # sentences ending with question mark
        r'[Ee]xplain\s+[^.!?]*\.', # Sentences starting with "Explain..."
        r'[Dd]evelop\s+[^.!?]*\.', # Sentences starting with "Develop..."
        r'[Dd]esign\s+[^.!?]*\.', # Sentences starting with "Design..."
        r'[Ww]rite\s+[^.!?]*\.', # Sentences starting with "Write..."
        r'[Ii]llustrate\s+[^.!?]*\.', # Sentences starting with "Illustrate..."
        r'[Dd]iscuss\s+[^.!?]*\.', # Sentences starting with "Discuss..."
        r'[Dd]escribe\s+[^.!?]*\.', # Sentences starting with "Describe..."
        r'[Ii]mplement\s+[^.!?]*\.', # Sentences starting with "Implement..."
        r'[Cc]reate\s+[^.!?]*\.', # Sentences starting with "Create..."
    ]
    
    questions = []
    
    for pattern in patterns:
        found = re.findall(pattern, text)
        for q in found:
            # Clean up the question
            q = q.strip()
            # Minimum length to filter out noise
            if len(q) > 15 and q not in questions:  
                questions.append(q)
    
    return questions

# Function to get model predictions
def get_predictions(model, questions):
    # Get predictions for extracted questions
    predictions = []
    
    for q in questions:
        prob = model.predict_probabilities(q)
        predictions.append({
            'question': q,
            'probability': prob
        })
    
    # Sort by probability
    predictions.sort(key=lambda x: x['probability'], reverse=True)
    return predictions

# Function to get probability interpretation
def get_probability_interpretation(probability):
    if probability >= 70:
        return "🔴 **Very High** - This question is very likely to appear!", "success"
    elif probability >= 50:
        return "🟡 **High** - This question has a good chance of appearing.", "warning"
    elif probability >= 30:
        return "🟠 **Medium** - This question might appear.", "info"
    else:
        return "🟢 **Low** - This question is less likely to appear.", "info"

# Cache the model to avoid retraining
@st.cache_resource
def get_trained_model():
    dataset = prepare_enhanced_dataset()
    model = EnhancedQuestionProbabilityModel(
        dataset['historical_questions'],
        dataset['appearance_counts'],
        dataset['recency_weights']
    )
    model.train(dataset['training_data'], epochs=10, learning_rate=0.001)
    return model, dataset

# Main Streamlit app
def main():
    st.title(" Exam Question Probability Predictor")
    
    st.write("""
    This tool predicts the likelihood of questions appearing in your next exam based on historical data.
    You can either upload files or enter questions manually.
    """)
    
    # Sidebar for additional information
    with st.sidebar:
        st.header("ℹ️ How it works")
        st.write("""
        1. **Historical Analysis**: Based on past exam patterns
        2. **AI Prediction**: Uses advanced ML algorithms
        3. **Probability Scoring**: 0-100% likelihood scale
        """)
        
        st.header("📊 Probability Scale")
        st.write("""
        - 🔴 **70-100%**: Very High
        - 🟡 **50-69%**: High  
        - 🟠 **30-49%**: Medium
        - 🟢 **0-29%**: Low
        """)
    
    # Create tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["📝 Enter Question", "📁 Upload Files", "📈 View Historical Data"])
    
    with tab1:
        st.subheader("Enter a Question Manually")
        
        # Text input for user to enter a question
        user_question = st.text_area(
            "Enter your question here:",
            placeholder="Type your question here... (e.g., 'Explain list comprehension in Python')",
            height=100,
            key="manual_question"
        )
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            predict_button = st.button("🔍 Predict Probability", type="primary")
        
        if predict_button:
            if user_question.strip():
                # Get trained model
                with st.spinner("Analyzing question..."):
                    model, dataset = get_trained_model()
                    probability = model.predict_probabilities(user_question.strip())
                
                # Display the result
                st.subheader("📊 Prediction Result")
                
                # Create columns for better layout
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.info(f"**Question:** {user_question}")
                    interpretation, alert_type = get_probability_interpretation(probability)
                    
                    if alert_type == "success":
                        st.success(f"**Probability:** {probability:.2f}%")
                        st.success(interpretation)
                    elif alert_type == "warning":
                        st.warning(f"**Probability:** {probability:.2f}%")
                        st.warning(interpretation)
                    else:
                        st.info(f"**Probability:** {probability:.2f}%")
                        st.info(interpretation)
                
                with col2:
                    # Create a simple gauge chart
                    st.metric(
                        label="Probability Score",
                        value=f"{probability:.1f}%",
                        delta=None
                    )
                
            else:
                st.error("⚠️ Please enter a question to predict its probability.")
    
    with tab2:
        st.subheader("Upload Files for Bulk Analysis")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload Word (.docx, .doc) or PDF files containing questions",
            type=["pdf", "docx", "doc"], 
            accept_multiple_files=True,
            key="file_uploader"
        )
        
        if uploaded_files:
            all_questions = []
            
            # Process all uploaded files
            for file in uploaded_files:
                st.write(f"📄 Processing: {file.name}")
                
                # Create a temporary file to handle the upload
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(file.getbuffer())
                    tmp_path = tmp_file.name
                
                # Extract text based on file type
                try:
                    if file.name.lower().endswith('.pdf'):
                        text = extract_text_from_pdf(tmp_path)
                    elif file.name.lower().endswith(('.docx', '.doc')):
                        text = extract_text_from_docx(tmp_path)
                    else:
                        st.error(f"Unsupported file format: {file.name}")
                        continue
                    
                    # Extract questions from text
                    questions = extract_questions(text)
                    
                    # Add to all questions
                    all_questions.extend(questions)
                    
                    st.success(f"✅ Extracted {len(questions)} questions from {file.name}")
                    
                except Exception as e:
                    st.error(f"❌ Error processing {file.name}: {str(e)}")
                
                # Clean up the temporary file
                os.unlink(tmp_path)
            
            if all_questions:
                # Remove duplicates
                all_questions = list(set(all_questions))
                
                # Get trained model and predictions
                with st.spinner("Training the prediction model and analyzing questions..."):
                    model, dataset = get_trained_model()
                    predictions = get_predictions(model, all_questions)
                
                # Display results
                st.subheader("📊 Question Prediction Results")
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                
                high_priority = len([p for p in predictions if p['probability'] >= 70])
                medium_priority = len([p for p in predictions if 50 <= p['probability'] < 70])
                low_priority = len([p for p in predictions if p['probability'] < 50])
                
                col1.metric("Total Questions", len(predictions))
                col2.metric("Very High Priority", high_priority)
                col3.metric("High Priority", medium_priority) 
                col4.metric("Medium/Low Priority", low_priority)
                
                # Top likely questions
                st.write("### 🔥 Top 10 Most Likely Questions:")
                
                # Create DataFrame for better display
                top_10 = predictions[:10]
                df = pd.DataFrame(top_10)
                df['probability'] = df['probability'].apply(lambda x: f"{x:.2f}%")
                df['priority'] = df.apply(lambda row: 
                    "🔴 Very High" if float(row['probability'].strip('%')) >= 70 
                    else "🟡 High" if float(row['probability'].strip('%')) >= 50
                    else "🟠 Medium" if float(row['probability'].strip('%')) >= 30
                    else "🟢 Low", axis=1)
                df.index = range(1, len(df) + 1)
                
                # Rename columns for better display
                df.columns = ['Question', 'Probability', 'Priority Level']
                st.dataframe(df, use_container_width=True)
                
                # Show all extracted questions
                with st.expander("📋 View All Extracted Questions"):
                    all_df = pd.DataFrame(predictions)
                    all_df['probability'] = all_df['probability'].apply(lambda x: f"{x:.2f}%")
                    all_df['priority'] = all_df.apply(lambda row: 
                        "🔴 Very High" if float(row['probability'].strip('%')) >= 70 
                        else "🟡 High" if float(row['probability'].strip('%')) >= 50
                        else "🟠 Medium" if float(row['probability'].strip('%')) >= 30
                        else "🟢 Low", axis=1)
                    all_df.index = range(1, len(all_df) + 1)
                    all_df.columns = ['Question', 'Probability', 'Priority Level']
                    st.dataframe(all_df, use_container_width=True)
                
                # Download option for results
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Top 10 Results as CSV",
                    data=csv,
                    file_name="top_10_exam_questions.csv",
                    mime="text/csv"
                )
    
    with tab3:
        st.subheader("📈 Historical Question Analysis")
        
        # Get the dataset for display
        model, dataset = get_trained_model()
        
        # Get base model predictions for comparison
        base_probs = model.predict_probabilities()
        sorted_base = sorted(base_probs.items(), key=lambda x: float(x[1].strip('%')), reverse=True)
        
        st.write("### 🏆 Top Questions Based on Historical Data:")
        
        historical_df = pd.DataFrame({
            'question': [q for q, _ in sorted_base[:15]],
            'probability': [p for _, p in sorted_base[:15]]
        })
        historical_df.index = range(1, len(historical_df) + 1)
        historical_df.columns = ['Question', 'Historical Probability']
        st.dataframe(historical_df, use_container_width=True)
        
        # Show dataset statistics
        st.write("### 📊 Dataset Statistics:")
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Total Historical Questions", len(dataset['historical_questions']))
        col2.metric("Average Appearance Count", f"{np.mean(dataset['appearance_counts']):.1f}")
        col3.metric("Most Frequent Count", max(dataset['appearance_counts']))
    
    # Quick question check at the bottom
    st.divider()
    st.subheader("⚡ Quick Question Check")
    st.write("Enter any question for a rapid probability assessment:")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        quick_question = st.text_input(
            "Question:",
            placeholder="e.g., What is inheritance in Python?",
            key="quick_question"
        )
    
    with col2:
        st.write("")  # Empty space for alignment
        check_button = st.button("🚀 Quick Check", key="quick_check")
    
    if check_button and quick_question.strip():
        with st.spinner("Analyzing..."):
            model, dataset = get_trained_model()
            probability = model.predict_probabilities(quick_question.strip())
            
            # Display result in a compact format
            interpretation, alert_type = get_probability_interpretation(probability)
            
            if alert_type == "success":
                st.success(f"**{probability:.1f}%** - {interpretation}")
            elif alert_type == "warning":
                st.warning(f"**{probability:.1f}%** - {interpretation}")
            else:
                st.info(f"**{probability:.1f}%** - {interpretation}")
    
    elif check_button and not quick_question.strip():
        st.error("⚠️ Please enter a question to check.")
    
    # Footer
    # st.divider()
    # st.markdown("""
    # <div style='text-align: center; color: gray;'>
    #     <small>📚 Exam Question Probability Predictor | Powered by AI & Historical Analysis</small>
    # </div>
    # """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()