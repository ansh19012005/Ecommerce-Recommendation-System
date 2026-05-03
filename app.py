from pathlib import Path
import sys

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

DATA_DIR = BASE_DIR / "data"
DB_PATH = BASE_DIR / "ecommerce.db"

RAW_DATA_PATH = DATA_DIR / "kz.csv"
CLEAN_DATA_PATH = DATA_DIR / "clean_data.csv"
TRENDING_DATA_PATH = DATA_DIR / "trending_products.csv"

ensure_processed_data(RAW_DATA_PATH, CLEAN_DATA_PATH, TRENDING_DATA_PATH)

clean_df = pd.read_csv(CLEAN_DATA_PATH)
clean_df.columns = clean_df.columns.str.lower()

trending_df = pd.read_csv(TRENDING_DATA_PATH)
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
    role = db.Column(db.String(20), nullable=False, default="user")


with app.app_context():
    db.create_all()


@app.route("/")
def index():
    if "user_id" not in session:
        return redirect(url_for("signin"))

    if session.get("role") == "admin":
        return redirect(url_for("analytics"))

    trending_products = get_trending_products(trending_df, limit=8)

    stats = {
        "total_users": int(clean_df["user_id"].nunique()),
        "total_products": int(clean_df["product_id"].nunique()),
        "total_categories": int(clean_df["category_code"].nunique()),
        "avg_price": round(float(clean_df["price"].mean()), 2),
    }

    return render_template(
        "index.html",
        trending_products=trending_products,
        stats=stats,
    )


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        fullname = request.form.get("fullname", "").strip()
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()

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
            role="user",
        )
        db.session.add(user)
        db.session.commit()

        flash("Account created successfully. Please sign in.", "success")
        return redirect(url_for("signin"))

    return render_template("signup.html")


@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "POST":
        login_type = request.form.get("login_type", "").strip()
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if not login_type or not username or not password:
            flash("Please fill all fields.", "danger")
            return redirect(url_for("signin"))

        user = User.query.filter_by(username=username).first()

        if not user or not check_password_hash(user.password_hash, password):
            flash("Invalid username or password.", "danger")
            return redirect(url_for("signin"))

        if user.role != login_type:
            flash(f"This account is not registered as {login_type}.", "danger")
            return redirect(url_for("signin"))

        session["user_id"] = user.id
        session["username"] = user.username
        session["role"] = user.role

        flash("Login successful!", "success")

        if user.role == "admin":
            return redirect(url_for("analytics"))
        return redirect(url_for("index"))

    return render_template("signin.html")


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        new_password = request.form.get("new_password", "").strip()

        if not username or not email or not new_password:
            flash("Please fill all fields.", "danger")
            return redirect(url_for("forgot_password"))

        user = User.query.filter_by(username=username, email=email).first()

        if not user:
            flash("No account found with provided username and email.", "danger")
            return redirect(url_for("forgot_password"))

        user.password_hash = generate_password_hash(new_password)
        db.session.commit()

        flash("Password reset successful. Please sign in.", "success")
        return redirect(url_for("signin"))

    return render_template("forgot_password.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "warning")
    return redirect(url_for("signin"))


@app.route("/main", methods=["GET", "POST"])
def main():
    if "user_id" not in session:
        flash("Please sign in first.", "warning")
        return redirect(url_for("signin"))

    if session.get("role") != "user":
        flash("Access denied. User login required.", "danger")
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
    if "user_id" not in session:
        flash("Please sign in first.", "warning")
        return redirect(url_for("signin"))

    if session.get("role") != "admin":
        flash("Access denied. Admin only.", "danger")
        return redirect(url_for("signin"))

    total_users = int(clean_df["user_id"].nunique())
    total_products = int(clean_df["product_id"].nunique())
    total_categories = int(clean_df["category_code"].nunique())
    avg_price = round(float(clean_df["price"].mean()), 2)

    top_brands_df = clean_df["brand"].value_counts().head(5).reset_index()
    top_brands_df.columns = ["brand", "count"]

    top_categories_df = clean_df["category_code"].value_counts().head(5).reset_index()
    top_categories_df.columns = ["category", "count"]

    # dummy / fallback market basket rules display
    market_basket_rules = []
    rules_path = BASE_DIR / "data" / "market_basket_rules.csv"
    if rules_path.exists():
        rules_df = pd.read_csv(rules_path)
        market_basket_rules = rules_df.head(5).to_dict(orient="records")

    return render_template(
        "analytics.html",
        total_users=total_users,
        total_products=total_products,
        total_categories=total_categories,
        avg_price=avg_price,
        top_brands=top_brands_df.to_dict(orient="records"),
        top_categories=top_categories_df.to_dict(orient="records"),
        market_basket_rules=market_basket_rules,
        username=session.get("username"),
    )


if __name__ == "__main__":
    app.run(debug=True)