import sys
import json
import pandas as pd
from pathlib import Path
from flask import render_template

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

import pandas as pd
from flask import Flask, flash, redirect, render_template, request, session, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash

from preprocessing import ensure_processed_data
from recommender import (
    content_based_recommendations,
    get_product_names,
    get_trending_products,
)
from collaborative import collaborative_filtering
from hybrid import hybrid_recommendation

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

CLEAN_DATA_PATH = DATA_DIR / "clean_data.csv"
TRENDING_PRODUCTS_PATH = DATA_DIR / "trending_products.csv"
SEGMENTS_PATH = DATA_DIR / "customer_segments.csv"
BASKET_RULES_PATH = DATA_DIR / "market_basket_rules.csv"
EVALUATION_PATH = DATA_DIR / "evaluation_metrics.json"

DATA_DIR = BASE_DIR / "data"
DB_PATH = BASE_DIR / "ecommerce.db"

SEGMENTS_PATH = DATA_DIR / "customer_segments.csv"
BASKET_RULES_PATH = DATA_DIR / "market_basket_rules.csv"
EVALUATION_PATH = DATA_DIR / "evaluation_metrics.json"

RAW_DATA_PATH = DATA_DIR / "kz.csv"
CLEAN_DATA_PATH = DATA_DIR / "clean_data.csv"
TRENDING_DATA_PATH = DATA_DIR / "trending_products.csv"

ensure_processed_data(RAW_DATA_PATH, CLEAN_DATA_PATH, TRENDING_DATA_PATH)

clean_df = pd.read_csv(CLEAN_DATA_PATH)
trending_df = pd.read_csv(TRENDING_PRODUCTS_PATH)

clean_df.columns = clean_df.columns.str.lower()
trending_df.columns = trending_df.columns.str.lower()

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "app" / "templates"),
    static_folder=str(BASE_DIR / "app" / "static"),
)

app.config["SECRET_KEY"] = "major_project_secret_key"
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fullname = db.Column(db.String(120), nullable=False)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)


with app.app_context():
    db.create_all()


@app.route("/")
def index():
    trending_products = get_trending_products(trending_df, limit=8)

    stats = {
        "total_users": int(clean_df["user_id"].nunique()),
        "total_products": int(clean_df["product_id"].nunique()),
        "total_categories": int(clean_df["category_code"].nunique()),
        "avg_price": round(float(clean_df["price"].mean()), 2)
    }

    return render_template(
        "index.html",
        trending_products=trending_products,
        stats=stats
    )


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        fullname = request.form.get("fullname", "").strip()
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()

        print("SIGNUP DATA:", fullname, username, email, password)

        if not fullname or not username or not email or not password:
            flash("Please fill all fields.", "danger")
            return redirect(url_for("signup"))

        existing_user = User.query.filter(
            (User.username == username) | (User.email == email)
        ).first()

        if existing_user:
            flash("Username or email already exists.", "danger")
            return redirect(url_for("signup"))

        hashed_password = generate_password_hash(password)

        user = User(
            fullname=fullname,
            username=username,
            email=email,
            password_hash=hashed_password,
        )
        db.session.add(user)
        db.session.commit()

        flash("Signup successful. Please sign in.", "success")
        return redirect(url_for("signin"))

    return render_template("signup.html")


@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        print("SIGNIN DATA:", username, password)

        user = User.query.filter_by(username=username).first()
        print("USER FOUND:", user)

        if user and check_password_hash(user.password_hash, password):
            session["user_id"] = user.id
            session["username"] = user.username
            flash("Login successful.", "success")
            return redirect(url_for("main"))

        flash("Invalid username or password.", "danger")
        return redirect(url_for("signin"))

    return render_template("signin.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("index"))


@app.route("/main", methods=["GET", "POST"])
def main():
    if "user_id" not in session:
        flash("Please sign in first.", "warning")
        return redirect(url_for("signin"))

    product_names = get_product_names(clean_df)
    recommendations = []

    if request.method == "POST":
        selected_product = request.form.get("product_name", "").strip()
        current_user_id = session["user_id"]

        content_recs = content_based_recommendations(clean_df, selected_product, top_n=8)
        collab_recs = collaborative_filtering(current_user_id, clean_df, top_n=8)
        recommendations = hybrid_recommendation(content_recs, collab_recs, top_n=8)

    return render_template(
        "main.html",
        product_names=product_names,
        recommendations=recommendations,
        username=session.get("username"),
    )


@app.route("/analytics")
def analytics():
    total_users = int(clean_df["user_id"].nunique())
    total_products = int(clean_df["product_id"].nunique())
    total_categories = int(clean_df["category_code"].nunique())
    avg_price = round(float(clean_df["price"].mean()), 2)
    total_revenue = round(float(clean_df["price"].sum()), 2)
    total_orders = int(len(clean_df))

    top_brands = clean_df["brand"].fillna("Unknown").value_counts().head(5).reset_index()
    top_brands.columns = ["brand", "count"]

    top_categories = clean_df["category_code"].fillna("Unknown").value_counts().head(5).reset_index()
    top_categories.columns = ["category", "count"]

    trending_products = []
    if not trending_df.empty:
        trending_products = (
            trending_df[["name", "brand", "interaction_count", "avg_rating"]]
            .head(5)
            .fillna("")
            .to_dict(orient="records")
        )

    segment_summary = []
    if SEGMENTS_PATH.exists():
        segments_df = pd.read_csv(SEGMENTS_PATH)
        if not segments_df.empty and "segment_name" in segments_df.columns:
            segment_summary = (
                segments_df["segment_name"]
                .value_counts()
                .reset_index()
            )
            segment_summary.columns = ["segment_name", "count"]
            segment_summary = segment_summary.to_dict(orient="records")

    basket_rules = []
    if BASKET_RULES_PATH.exists():
        basket_df = pd.read_csv(BASKET_RULES_PATH)
        if not basket_df.empty:
            basket_df = basket_df.sort_values(by=["lift", "confidence"], ascending=False).head(5)
            basket_rules = basket_df[
                ["antecedents", "consequents", "support", "confidence", "lift"]
            ].to_dict(orient="records")

    evaluation = {}
    if EVALUATION_PATH.exists():
        with open(EVALUATION_PATH, "r", encoding="utf-8") as f:
            evaluation = json.load(f)

    rec_metrics = evaluation.get("recommendation", {})
    seg_metrics = evaluation.get("segmentation", {})
    business_metrics = evaluation.get("business_metrics", {})

    brand_chart_labels = top_brands["brand"].tolist()
    brand_chart_values = top_brands["count"].tolist()

    category_chart_labels = top_categories["category"].tolist()
    category_chart_values = top_categories["count"].tolist()

    segment_chart_labels = [item["segment_name"] for item in segment_summary] if segment_summary else []
    segment_chart_values = [item["count"] for item in segment_summary] if segment_summary else []

    trending_chart_labels = [item["name"] for item in trending_products] if trending_products else []
    trending_chart_values = [item["interaction_count"] for item in trending_products] if trending_products else []

    return render_template(
        "analytics.html",
        total_users=total_users,
        total_products=total_products,
        total_categories=total_categories,
        avg_price=avg_price,
        total_revenue=total_revenue,
        total_orders=total_orders,
        top_brands=top_brands.to_dict(orient="records"),
        top_categories=top_categories.to_dict(orient="records"),
        trending_products=trending_products,
        segment_summary=segment_summary,
        basket_rules=basket_rules,
        rec_metrics=rec_metrics,
        seg_metrics=seg_metrics,
        business_metrics=business_metrics,
        brand_chart_labels=brand_chart_labels,
        brand_chart_values=brand_chart_values,
        category_chart_labels=category_chart_labels,
        category_chart_values=category_chart_values,
        segment_chart_labels=segment_chart_labels,
        segment_chart_values=segment_chart_values,
        trending_chart_labels=trending_chart_labels,
        trending_chart_values=trending_chart_values,
    )


if __name__ == "__main__":
    app.run(debug=True)