import re
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI Career Assistant",
    page_icon="🎯",
    layout="wide"
)

ROLE_DATA = {
    "Python Developer": {
        "skills": ["python", "sql", "git", "api", "flask", "django", "oop", "data structures"],
        "description": "Build backend applications, APIs and automation tools using Python.",
        "roadmap": ["Python & OOP", "SQL", "Git/GitHub", "REST APIs", "Flask/Django", "DSA & Projects"]
    },
    "Data Analyst": {
        "skills": ["python", "sql", "excel", "pandas", "numpy", "statistics", "power bi", "tableau", "data visualization"],
        "description": "Analyze data and communicate insights through dashboards and reports.",
        "roadmap": ["Excel", "SQL", "Statistics", "Python/Pandas", "Power BI/Tableau", "Portfolio Projects"]
    },
    "Data Scientist": {
        "skills": ["python", "sql", "pandas", "numpy", "statistics", "machine learning", "scikit-learn", "matplotlib", "deep learning"],
        "description": "Use statistics and machine learning to solve prediction and classification problems.",
        "roadmap": ["Python", "Statistics", "Pandas/NumPy", "Machine Learning", "Model Evaluation", "ML Projects"]
    },
    "Machine Learning Engineer": {
        "skills": ["python", "machine learning", "scikit-learn", "tensorflow", "pytorch", "sql", "git", "docker", "apis"],
        "description": "Develop, deploy and maintain machine-learning models and pipelines.",
        "roadmap": ["Python & ML", "Deep Learning", "APIs", "Docker", "Cloud Basics", "End-to-End ML Projects"]
    },
    "Web Developer": {
        "skills": ["html", "css", "javascript", "react", "git", "api", "node.js", "sql", "responsive design"],
        "description": "Create responsive websites and modern web applications.",
        "roadmap": ["HTML/CSS", "JavaScript", "Git/GitHub", "React", "Backend/API", "Full-Stack Project"]
    },
    "Cybersecurity Analyst": {
        "skills": ["networking", "linux", "python", "cybersecurity", "siem", "cryptography", "ethical hacking", "sql"],
        "description": "Monitor systems, investigate threats and improve security controls.",
        "roadmap": ["Networking", "Linux", "Security Fundamentals", "SIEM", "Ethical Hacking", "Security Labs"]
    },
}

ALIASES = {
    "js": "javascript",
    "powerbi": "power bi",
    "ml": "machine learning",
    "scikit learn": "scikit-learn",
    "node": "node.js",
    "reactjs": "react",
}

def normalize(text):
    text = text.lower()
    for old, new in ALIASES.items():
        text = text.replace(old, new)
    return re.sub(r"[^a-z0-9+#.\- ]", " ", text)

def extract_skills(text):
    clean = normalize(text)
    found = []
    all_skills = sorted({s for role in ROLE_DATA.values() for s in role["skills"]}, key=len, reverse=True)
    for skill in all_skills:
        if skill in clean and skill not in found:
            found.append(skill)
    return found

def recommend_roles(user_text):
    documents = [user_text] + [" ".join(v["skills"]) for v in ROLE_DATA.values()]
    matrix = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(documents)
    scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    roles = list(ROLE_DATA.keys())
    return sorted(zip(roles, scores), key=lambda x: x[1], reverse=True)

st.title("🎯 AI Career Assistant")
st.write("An NLP-based career recommendation and skill-gap analysis tool for students and job seekers.")

st.info(
    "How it works: the assistant extracts technical skills from your profile, "
    "compares them with predefined career-role profiles using TF-IDF + cosine similarity, "
    "then generates role recommendations, skill gaps and a learning roadmap."
)

with st.sidebar:
    st.header("Your Profile")
    name = st.text_input("Name", placeholder="Your name")
    education = st.selectbox(
        "Education",
        ["BCA", "B.Tech / BE", "MCA", "B.Sc", "M.Tech", "Other"]
    )
    experience = st.selectbox(
        "Experience",
        ["Student / Fresher", "0-1 years", "1-3 years", "3+ years"]
    )

profile = st.text_area(
    "Describe your skills, projects and interests",
    height=180,
    placeholder=(
        "Example: I am a BCA student with Python, SQL, HTML, CSS, Git and "
        "Pandas. I have built a library management system and a data analysis project."
    )
)

if st.button("Analyze My Career", type="primary", use_container_width=True):
    if len(profile.strip()) < 15:
        st.warning("Please enter a little more information about your skills and projects.")
        st.stop()

    normalized_profile = normalize(profile)
    skills = extract_skills(profile)
    recommendations = recommend_roles(normalized_profile)

    st.session_state["result"] = {
        "skills": skills,
        "recommendations": recommendations
    }

if "result" in st.session_state:
    result = st.session_state["result"]
    skills = result["skills"]
    recommendations = result["recommendations"]

    st.divider()
    st.header(f"Career Analysis{' for ' + name if name else ''}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Skills Detected", len(skills))
    c2.metric("Top Career Match", recommendations[0][0])
    c3.metric("Match Score", f"{recommendations[0][1] * 100:.1f}%")

    st.subheader("🧠 Recommended Career Paths")

    rec_df = pd.DataFrame(
        [
            {"Career": role, "Match": round(score * 100, 1)}
            for role, score in recommendations
        ]
    )

    fig = px.bar(
        rec_df,
        x="Match",
        y="Career",
        orientation="h",
        range_x=[0, 100],
        labels={"Match": "Match Score (%)"}
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    top_role = recommendations[0][0]
    role_info = ROLE_DATA[top_role]

    left, right = st.columns(2)

    with left:
        st.subheader(f"🎯 Best Match: {top_role}")
        st.write(role_info["description"])

        st.markdown("**Skills detected in your profile:**")
        if skills:
            st.write(", ".join(skill.title() for skill in skills))
        else:
            st.write("No predefined technical skills detected.")

    with right:
        st.subheader("📚 Skill Gap")
        missing = [s for s in role_info["skills"] if s not in skills]

        if missing:
            for skill in missing:
                st.write(f"- {skill.title()}")
        else:
            st.success("You already cover the main skills for this role!")

    st.subheader("🗺️ Suggested Learning Roadmap")
    for i, item in enumerate(role_info["roadmap"], 1):
        st.write(f"**{i}.** {item}")

    st.subheader("💡 Personalized Advice")
    if missing:
        st.write(
            f"Your strongest next step is to focus on **{missing[0].title()}** and build "
            f"a small project around it. Then add the project to GitHub as portfolio evidence."
        )
    else:
        st.write(
            f"Your profile is already aligned with **{top_role}**. Focus on stronger projects, "
            "interview preparation and real-world deployment."
        )

    st.caption(
        "Note: This is an educational AI/NLP recommendation system, not professional career advice. "
        "The model uses the skills and role profiles included in this project."
    )
else:
    st.subheader("What you get")
    a, b, c = st.columns(3)
    a.write("**1. Skill Extraction**\n\nDetects technical skills from your profile.")
    b.write("**2. Career Matching**\n\nRanks career paths using NLP similarity.")
    c.write("**3. Skill Gap & Roadmap**\n\nShows missing skills and a learning path.")
