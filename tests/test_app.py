import uuid

from app import app


def test_home_redirects_to_login_when_not_authenticated():
    client = app.test_client()

    response = client.get("/", follow_redirects=False)

    assert response.status_code == 302
    assert response.headers["Location"] == "/login"


def test_registration_succeeds_with_matching_passwords():
    client = app.test_client()
    unique_email = f"testuser_{uuid.uuid4().hex[:8]}@example.com"

    response = client.post(
        "/register",
        data={
            "fullname": "Test User",
            "email": unique_email,
            "password": "StrongPass123",
            "confirm_password": "StrongPass123",
        },
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"Registration successful" in response.data
