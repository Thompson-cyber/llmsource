"""
Module containing authentication logic
"""

import bcrypt
import functools
from flask import g, redirect, url_for


"""
Function to hash a password
This function was generated by ChatGPT from the prompt: "python password hashing using bcrypt"
"""
def hash_password(password):
    # Generate a salt and hash the password
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed_password

"""
Function verify a password
This function was generated by ChatGPT from the prompt: "python password hashing using bcrypt"
"""
def verify_password(password, hashed_password):
    # Check if the provided password matches the hashed password
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password)


"""
Decorator to require a user to be logged in
Generated by GitHub Copilot
"""
def authenticate(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None and g.token is None:
            # Redirect to the login page
            return redirect(url_for("login"))
        if g.token:
            # Validate the token
            if not validate_token(g.token):
                # Redirect to the login page
                return redirect(url_for("login"))
        return view(**kwargs)
    return wrapped_view

def validate_token(token):
    """
    Mock function to validate a token
    """
    return token