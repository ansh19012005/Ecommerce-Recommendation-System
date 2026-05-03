import importlib.util
from pathlib import Path
from werkzeug.security import generate_password_hash

project_root = Path(__file__).resolve().parent
app_file = project_root / "app.py"

spec = importlib.util.spec_from_file_location("main_app", app_file)
main_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main_app)

app = main_app.app
db = main_app.db
User = main_app.User

with app.app_context():
    existing = User.query.filter_by(username="admin").first()
    if existing:
        print("Admin already exists.")
    else:
        admin = User(
            fullname="Administrator",
            username="admin",
            email="admin@smartshop.com",
            password_hash=generate_password_hash("admin123"),
            role="admin"
        )
        db.session.add(admin)
        db.session.commit()
        print("Admin created successfully.")