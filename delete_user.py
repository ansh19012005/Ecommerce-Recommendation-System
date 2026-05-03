from app import db, User
import app   # this imports app.py

with app.app.app_context():
    username_to_delete = "xyz@11"

    user = User.query.filter_by(username=username_to_delete).first()

    if user:
        db.session.delete(user)
        db.session.commit()
        print(f"User '{username_to_delete}' deleted successfully.")
    else:
        print("User not found.")