from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, Length, EqualTo, ValidationError

# Mock User model for demonstration. Replace with your actual User model import.
# from .models import User 

class RequestResetForm(FlaskForm):
    """
    Form for users to request a password reset by submitting their email.
    """
    email = StringField('Email',
                        validators=[DataRequired(), Email()])
    submit = SubmitField('Request Password Reset')

    # Optional: Add validation to ensure the email exists in the database
    # def validate_email(self, email):
    #     user = User.query.filter_by(email=email.data).first()
    #     if user is None:
    #         raise ValidationError('There is no account with that email. You must register first.')


class ResetPasswordForm(FlaskForm):
    """
    Form for users to set a new password.
    """
    password = PasswordField('New Password', 
                             validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm New Password',
                                     validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Reset Password')
