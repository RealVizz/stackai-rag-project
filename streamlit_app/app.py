import streamlit as st

from streamlit_app.state import initialize_session_state
from streamlit_app.views import (
    page_welcome,
    page_user_dashboard,
    page_admin_dashboard,
    page_chat_window
)

# --- Initialize Session State ---
# This is the first and most crucial step. The `initialize_session_state` function,
# defined in `state.py`, sets up all the necessary variables in Streamlit's session state.
# This ensures that variables persist across reruns of the script, which happens on every
# user interaction. It initializes everything from user roles and chat data to widget states.
initialize_session_state()


# --- Main Application Router ---
def main():
    """
    This main function acts as a router to control which page is displayed to the user.
    The routing is based on the 'role' and other view-related flags stored in the session state.
    
    - If no role is selected (`st.session_state.role is None`), it shows the welcome page.
    - If the role is 'user', it decides whether to show the user dashboard or the chat window
      based on the `user_view_chat` flag.
    - If the role is 'admin', it similarly routes between the admin dashboard and the chat window
      based on the `view_chat` flag.
    
    Back buttons in the chat window pages reset the respective view flags to return to the dashboards.
    """

    st.set_page_config(page_title="RAG Chatbot", layout="centered")

    # Welcome page is shown if no role is selected yet.
    if st.session_state.role is None:
        page_welcome()

    # User-specific routing
    elif st.session_state.role == "user":
        # If the `user_view_chat` flag is True, show the chat window.
        if st.session_state.user_view_chat:
            page_chat_window(is_admin_view=False)
            # Button to navigate back to the user dashboard.
            if st.button("Back to User Dashboard", key="back_to_user_dashboard_from_chat"):
                st.session_state.user_view_chat = False
                st.session_state.selected_chat_id = ""
                st.session_state.selected_chat_id_input_widget = ""  # Clear widget state
                st.session_state.chat_history = []
                st.rerun()
        # Otherwise, show the user dashboard.
        else:
            page_user_dashboard()

    # Admin-specific routing
    elif st.session_state.role == "admin":
        # If the `view_chat` flag is True, show the chat window in admin view.
        if st.session_state.view_chat:
            page_chat_window(is_admin_view=True)
            # Button to navigate back to the admin dashboard.
            if st.button("Back to Admin Dashboard", key="back_to_admin_dashboard_from_chat"):
                st.session_state.view_chat = False
                st.session_state.admin_selected_chat_id = ""
                st.session_state.admin_selected_chat_id_input_widget = ""  # Clear widget state
                st.session_state.chat_history = []
                st.rerun()
        # Otherwise, show the admin dashboard.
        else:
            page_admin_dashboard()


if __name__ == "__main__":
    # The main function is called to start the Streamlit application.
    main()
