import streamlit as st

from streamlit_app.utils import (
    get_all_chat_ids_for_user,
    get_chat_history,
    query_backend,
    upload_pdf_to_backend,
    get_all_user_ids,
)

st.set_page_config(page_title="RAG Chatbot", layout="centered")

# --- Session State Initialization ---
if "role" not in st.session_state:
    st.session_state.role = None
if "user_id" not in st.session_state:
    st.session_state.user_id = ""
if "chat_ids" not in st.session_state:
    st.session_state.chat_ids = []
if "selected_chat_id" not in st.session_state:
    st.session_state.selected_chat_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "new_chat_name" not in st.session_state:
    st.session_state.new_chat_name = ""
if "uploaded_files_info" not in st.session_state:
    st.session_state.uploaded_files_info = {}

# Admin specific session state
if "all_user_ids" not in st.session_state:
    st.session_state.all_user_ids = []
if "admin_selected_user_id" not in st.session_state:
    st.session_state.admin_selected_user_id = ""
if "admin_user_chat_ids" not in st.session_state:
    st.session_state.admin_user_chat_ids = []
if "admin_selected_chat_id" not in st.session_state:
    st.session_state.admin_selected_chat_id = None


# --- Helper Functions for UI Rendering ---
def render_welcome_screen():
    st.title("Welcome to the RAG Chatbot!")
    st.write("Please select your role to continue.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("I am a User", use_container_width=True):
            st.session_state.role = "user"
    with col2:
        if st.button("I am an Admin", use_container_width=True):
            st.session_state.role = "admin"

    st.info("Hint: Select your role to proceed to the respective dashboard.")


def render_chat_window(is_admin_view: bool = False):
    user_id = st.session_state.user_id if not is_admin_view else st.session_state.admin_selected_user_id
    chat_id = st.session_state.selected_chat_id if not is_admin_view else st.session_state.admin_selected_chat_id

    st.subheader(f"Chat with {chat_id} (User: {user_id})")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Display uploaded files
    current_chat_files = st.session_state.uploaded_files_info.get(chat_id, [])
    if current_chat_files:
        st.sidebar.subheader("Uploaded Files")
        for file_info in current_chat_files:
            status_icon = "✅" if file_info["status"] == "success" else "❌"
            st.sidebar.write(f"{status_icon} {file_info["filename"]}")

    if not is_admin_view:
        # Chat input for user
        user_query = st.chat_input(
            "Type your message here...", key="chat_input", disabled=st.session_state.get("file_uploader_active", False)
        )

        if user_query:
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.spinner("Thinking..."):
                response = query_backend(
                    st.session_state.user_id,
                    st.session_state.selected_chat_id,
                    user_query,
                )
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)

        # File uploader for user
        with st.sidebar:
            st.subheader("Upload PDF Files")
            uploaded_files = st.file_uploader(
                "Choose PDF files",
                type="pdf",
                accept_multiple_files=True,
                key="pdf_uploader",
                disabled=bool(user_query),  # Disable if chat input has content
                on_change=lambda: st.session_state.__setitem__("file_uploader_active",
                                                               True) if st.session_state.pdf_uploader else st.session_state.__setitem__(
                    "file_uploader_active", False)
            )

            if uploaded_files and st.button("Upload Selected Files", key="upload_button"):
                with st.spinner("Uploading and processing files..."):
                    upload_results = upload_pdf_to_backend(
                        st.session_state.user_id,
                        st.session_state.selected_chat_id,
                        uploaded_files,
                    )
                    if chat_id not in st.session_state.uploaded_files_info:
                        st.session_state.uploaded_files_info[chat_id] = []
                    st.session_state.uploaded_files_info[chat_id].extend(upload_results)
                    st.success("Files processed!")
                st.rerun()

        st.info("Hint: Type your question in the chat box or upload PDF files using the sidebar.")
    else:
        st.info("Admin View: You can only view this chat. Interaction is disabled.")


def render_user_dashboard():
    st.header("User Dashboard")

    # User ID input
    st.session_state.user_id = st.text_input(
        "Enter your User ID:", value=st.session_state.user_id, key="user_id_input"
    )

    # Fetch Chats button
    if st.button("Fetch My Chats", key="fetch_chats_button"):
        if st.session_state.user_id:
            st.session_state.chat_ids = get_all_chat_ids_for_user(
                st.session_state.user_id
            )
            if not st.session_state.chat_ids:
                st.info("You don't have any chats yet. Start a new one!")
            else:
                st.success(f"Found {len(st.session_state.chat_ids)} chats.")
        else:
            st.warning("Please enter a User ID.")

    # Existing Chats selection
    if st.session_state.chat_ids:
        selected_chat_option = st.selectbox(
            "Select an existing chat:",
            ["--- Select a chat ---"] + st.session_state.chat_ids,
            key="chat_selector",
        )
        if selected_chat_option != "--- Select a chat ---":
            st.session_state.selected_chat_id = selected_chat_option
            # Load chat history for the selected chat
            st.session_state.chat_history = get_chat_history(
                st.session_state.user_id, st.session_state.selected_chat_id
            )
            st.rerun()  # Rerun to switch to chat window

    st.markdown("--- ")

    # Start New Chat
    st.subheader("Start a New Chat")
    st.session_state.new_chat_name = st.text_input(
        "Enter a unique name for your new chat:", key="new_chat_name_input"
    )
    if st.button("Start New Chat", key="start_new_chat_button"):
        if st.session_state.user_id and st.session_state.new_chat_name:
            if st.session_state.new_chat_name in st.session_state.chat_ids:
                st.error("Chat name already exists. Please choose a unique name.")
            else:
                st.session_state.selected_chat_id = st.session_state.new_chat_name
                st.session_state.chat_ids.append(st.session_state.new_chat_name)
                st.session_state.chat_history = []  # Clear history for new chat
                st.rerun()  # Rerun to switch to chat window
        else:
            st.warning("Please enter both a User ID and a New Chat Name.")

    st.markdown("--- ")

    if st.button("Back to Role Selection", key="back_to_role_user"):
        st.session_state.role = None
        st.session_state.user_id = ""
        st.session_state.chat_ids = []
        st.session_state.selected_chat_id = None
        st.session_state.new_chat_name = ""
        st.session_state.uploaded_files_info = {}


def render_admin_dashboard():
    st.header("Admin Dashboard")

    if st.button("Fetch All Users", key="fetch_all_users_admin"):
        st.session_state.all_user_ids = get_all_user_ids()
        if not st.session_state.all_user_ids:
            st.info("No users found yet.")
        else:
            st.success(f"Found {len(st.session_state.all_user_ids)} users.")

    # Admin selects user from dropdown or enters custom ID
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        # Use a temporary variable to hold the selected option from selectbox
        selected_user_from_dropdown = st.selectbox(
            "Select a User ID:",
            ["--- Select a user ---"] + st.session_state.all_user_ids,
            key="admin_user_selector",
        )
        # Update the text input based on dropdown selection
        if selected_user_from_dropdown != "--- Select a user ---":
            st.session_state.admin_selected_user_id = selected_user_from_dropdown

        st.session_state.admin_selected_user_id = st.text_input(
            "Or enter User ID to view (e.g., if not in list):",
            value=st.session_state.admin_selected_user_id,
            key="admin_custom_user_id_input",
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacer
        if st.button("Fetch User's Chats", key="admin_fetch_user_chats"):
            if st.session_state.admin_selected_user_id:
                st.session_state.admin_user_chat_ids = get_all_chat_ids_for_user(
                    st.session_state.admin_selected_user_id
                )
                if not st.session_state.admin_user_chat_ids:
                    st.info(f"User '{st.session_state.admin_selected_user_id}' has no chats.")
                else:
                    st.success(
                        f"Found {len(st.session_state.admin_user_chat_ids)} chats for user '{st.session_state.admin_selected_user_id}'.")
            else:
                st.warning("Please select or enter a User ID.")

    # Admin selects chat from dropdown
    if st.session_state.admin_user_chat_ids:
        selected_admin_chat_option = st.selectbox(
            "Select a Chat ID to view:",
            ["--- Select a chat ---"] + st.session_state.admin_user_chat_ids,
            key="admin_chat_selector",
        )
        if selected_admin_chat_option != "--- Select a chat ---":
            st.session_state.admin_selected_chat_id = selected_admin_chat_option
            # Load chat history for the selected chat
            st.session_state.chat_history = get_chat_history(
                st.session_state.admin_selected_user_id,
                st.session_state.admin_selected_chat_id,
            )
            st.rerun()  # Rerun to switch to admin chat view

    st.markdown("--- ")

    if st.button("Back to Role Selection", key="back_to_role_admin"):
        st.session_state.role = None
        st.session_state.all_user_ids = []
        st.session_state.admin_selected_user_id = ""
        st.session_state.admin_user_chat_ids = []
        st.session_state.admin_selected_chat_id = None


# --- Main App Logic ---
if st.session_state.role is None:
    render_welcome_screen()
elif st.session_state.role == "user":
    if st.session_state.selected_chat_id:
        render_chat_window()
    else:
        render_user_dashboard()
elif st.session_state.role == "admin":
    if st.session_state.admin_selected_chat_id:
        render_chat_window(is_admin_view=True)  # Use the same chat window, but in read-only mode
        if st.button("Back to Admin Dashboard", key="back_to_admin_dashboard_from_chat"):
            st.session_state.admin_selected_chat_id = None
            st.rerun()
    else:
        render_admin_dashboard()
