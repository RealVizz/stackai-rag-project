import streamlit as st

from streamlit_app.state import (
    _clear_session_state_for_role_selection,
    _update_user_chat_selection_callback,
    _update_admin_user_selection_callback,
    _update_admin_chat_selection_callback,
    _update_user_id_from_input,
    _update_chat_id_from_input,
    _update_new_chat_name_from_input,
    _update_admin_user_id_from_input,
    _update_admin_chat_id_from_input,
    _file_uploader_on_change_callback,
    _handle_file_upload,
    _sync_user_inputs,
    _sync_admin_inputs
)
from streamlit_app.utils import (
    get_chat_history,
    query_backend,
    get_all_chat_ids_for_user,
    get_all_user_ids,
    get_uploaded_files
)


# --- Page: Welcome ---

def page_welcome():
    """
    Renders the initial role selection screen for the application.
    """
    st.title("Welcome to the RAG Chatbot!")
    st.write("Please select your role to continue.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("I am a User", use_container_width=True):
            st.session_state.role = "user"
            st.rerun()
    with col2:
        if st.button("I am an Admin", use_container_width=True):
            st.session_state.role = "admin"
            st.rerun()
    st.info("Hint: Select your role to proceed to the respective dashboard.")


# --- Page: Chat Window ---

def page_chat_window(is_admin_view: bool = False):
    """
    Renders the main chat interface for a user or admin.
    """
    user_id = st.session_state.user_id if not is_admin_view else st.session_state.admin_selected_user_id
    chat_id = st.session_state.selected_chat_id if not is_admin_view else st.session_state.admin_selected_chat_id

    if not user_id or not chat_id:
        st.warning("Please select a User ID and Chat ID to view messages.")
        if st.button("Go Back"):
            if is_admin_view:
                st.session_state.view_chat = False
                st.session_state.admin_selected_chat_id = ""
                st.session_state.admin_selected_chat_id_input_widget = ""
            else:
                st.session_state.user_view_chat = False
                st.session_state.selected_chat_id = ""
                st.session_state.selected_chat_id_input_widget = ""
            st.session_state.chat_history = []
            st.rerun()
        return

    st.subheader(f"Chat with {chat_id} (User: {user_id})")

    # --- Data Loading ---
    # Load chat history if it's not already in the session state.
    if not st.session_state.chat_history:
        st.session_state.chat_history = get_chat_history(user_id, chat_id)

    # Load the list of uploaded files if it's not already in the session state for this chat.
    if chat_id not in st.session_state.uploaded_files_info:
        st.session_state.uploaded_files_info[chat_id] = get_uploaded_files(user_id, chat_id)

    # --- Main Chat Display ---
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Sidebar Display ---
    with st.sidebar:
        st.header("Chat Information")
        # Display the list of uploaded files for the current chat.
        st.subheader("Uploaded Files")
        uploaded_files = st.session_state.uploaded_files_info.get(chat_id, [])
        if uploaded_files:
            for file_info in uploaded_files:
                # The backend now returns a simple dict with just 'filename'.
                st.success(file_info.get("filename", ""))
        else:
            st.info("No files have been uploaded to this chat yet.")

    # --- User/Admin Specific Components ---
    if not is_admin_view:
        # User View: Enable chat input and file uploading.
        user_query = st.chat_input(
            "Type your message here...",
            key="chat_input",
            disabled=st.session_state.file_uploader_active,
        )
        if st.session_state.file_uploader_active and not user_query:
            st.info("Files are ready to upload. Use the 'Upload' button or clear them to chat.")

        if user_query:
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.spinner("Thinking..."):
                response = query_backend(user_id, chat_id, user_query)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()

        with st.sidebar:
            st.markdown("--- ")
            st.subheader("Upload New PDF Files")
            current_uploader_key = f"pdf_uploader_{st.session_state.uploader_key}"
            st.file_uploader(
                "Choose PDF files",
                type="pdf",
                accept_multiple_files=True,
                key=current_uploader_key,
                on_change=_file_uploader_on_change_callback,
            )
            upload_button_disabled = not st.session_state.file_uploader_active
            st.button(
                "Upload Selected Files",
                key="upload_button",
                disabled=upload_button_disabled,
                on_click=_handle_file_upload
            )
    else:
        # Admin View: Disable interactions.
        st.info("Admin View: You can only view this chat. Interaction is disabled.")
        st.chat_input("Admin view: Chat is disabled.", disabled=True)


# --- Page: User Dashboard ---

def page_user_dashboard():
    """
    Renders the main dashboard for a regular user.
    """
    st.header("User Dashboard")

    st.text_input(
        "Enter your User ID:",
        key="user_id_input_widget",
        on_change=_update_user_id_from_input,
    )

    if st.button("Fetch My Chats", key="fetch_chats_button"):
        _sync_user_inputs()
        if st.session_state.user_id:
            with st.spinner("Fetching chats..."):
                st.session_state.chat_ids = get_all_chat_ids_for_user(st.session_state.user_id)
            if not st.session_state.chat_ids:
                st.info("You don't have any chats yet. Start a new one!")
            else:
                st.success(f"Found {len(st.session_state.chat_ids)} chats.")
        else:
            st.warning("Please enter a User ID.")

    if st.session_state.chat_ids or st.session_state.selected_chat_id:
        current_chat_index = 0
        if st.session_state.selected_chat_id in st.session_state.chat_ids:
            current_chat_index = st.session_state.chat_ids.index(st.session_state.selected_chat_id) + 1

        st.selectbox(
            "Select an existing chat:",
            ["--- Select a chat ---"] + st.session_state.chat_ids,
            key="chat_selector",
            index=current_chat_index,
            on_change=_update_user_chat_selection_callback,
        )
        st.text_input(
            "Or enter Chat ID to view (e.g., if not in list):",
            key="selected_chat_id_input_widget",
            on_change=_update_chat_id_from_input,
        )
        if st.button("Enter Chat", key="enter_chat_button"):
            _sync_user_inputs()
            if not st.session_state.user_id:
                st.warning("Please enter a User ID.")
            elif not st.session_state.selected_chat_id:
                st.warning("Please select or enter a Chat ID.")
            elif st.session_state.selected_chat_id not in st.session_state.chat_ids:
                st.error(f"Chat ID '{st.session_state.selected_chat_id}' not found for user. Please fetch chats first.")
            else:
                st.session_state.user_view_chat = True
                st.rerun()

    st.markdown("--- ")
    st.subheader("Start a New Chat")
    st.text_input(
        "Enter a unique name for your new chat:",
        key="new_chat_name_input_widget",
        on_change=_update_new_chat_name_from_input
    )
    if st.button("Start New Chat", key="start_new_chat_button"):
        _sync_user_inputs()
        if st.session_state.user_id and st.session_state.new_chat_name:
            if st.session_state.new_chat_name in st.session_state.chat_ids:
                st.error("Chat name already exists. Please choose a unique name.")
            else:
                new_chat_name = st.session_state.new_chat_name
                st.session_state.selected_chat_id = new_chat_name
                st.session_state.chat_ids.append(new_chat_name)
                st.session_state.chat_history = []
                st.session_state.new_chat_name = ""
                st.session_state.user_view_chat = True
                st.rerun()
        else:
            st.warning("Please enter both a User ID and a New Chat Name.")

    st.markdown("--- ")
    if st.button("Back to Role Selection", key="back_to_role_user"):
        _clear_session_state_for_role_selection()
        st.rerun()


# --- Page: Admin Dashboard ---

def page_admin_dashboard():
    """Renders the dashboard for the admin to view user chats."""
    st.header("Admin Dashboard")

    fetch_users_col, user_dropdown_col = st.columns([0.3, 0.7])
    with fetch_users_col:
        if st.button("Fetch All Users", key="fetch_all_users_admin", use_container_width=True):
            with st.spinner("Fetching all users..."):
                st.session_state.all_user_ids = get_all_user_ids()
            if not st.session_state.all_user_ids:
                st.info("No users found yet.")
            else:
                st.success(f"Found {len(st.session_state.all_user_ids)} users.")
                if st.session_state.admin_selected_user_id not in st.session_state.all_user_ids:
                    st.session_state.admin_selected_user_id = ""
                    st.session_state.admin_user_id_input_widget = ""
    with user_dropdown_col:
        current_user_index = 0
        if st.session_state.admin_selected_user_id in st.session_state.all_user_ids:
            current_user_index = st.session_state.all_user_ids.index(st.session_state.admin_selected_user_id) + 1
        st.selectbox(
            "Select a User ID:",
            ["--- Select a user ---"] + st.session_state.all_user_ids,
            key="admin_user_selector",
            index=current_user_index,
            on_change=_update_admin_user_selection_callback,
            label_visibility="collapsed",
        )
    user_id_input_col, fetch_chats_col = st.columns([0.7, 0.3])
    with user_id_input_col:
        st.text_input(
            "Or enter User ID to view (e.g., if not in list):",
            key="admin_user_id_input_widget",
            on_change=_update_admin_user_id_from_input,
        )
    with fetch_chats_col:
        st.markdown("<br>", unsafe_allow_html=True)
        fetch_chats_disabled = not st.session_state.admin_selected_user_id
        if st.button("Fetch User's Chats", key="admin_fetch_user_chats", use_container_width=True,
                     disabled=fetch_chats_disabled):
            _sync_admin_inputs()
            with st.spinner("Fetching user's chats..."):
                st.session_state.admin_user_chat_ids = get_all_chat_ids_for_user(
                    st.session_state.admin_selected_user_id)
            if not st.session_state.admin_user_chat_ids:
                st.info(f"User '{st.session_state.admin_selected_user_id}' has no chats.")
                st.session_state.admin_selected_chat_id = ""
                st.session_state.admin_selected_chat_id_input_widget = ""
            else:
                st.success(f"Found {len(st.session_state.admin_user_chat_ids)} chats for user.")

    if st.session_state.admin_user_chat_ids or st.session_state.admin_selected_chat_id:
        current_chat_index = 0
        if st.session_state.admin_selected_chat_id in st.session_state.admin_user_chat_ids:
            current_chat_index = st.session_state.admin_user_chat_ids.index(st.session_state.admin_selected_chat_id) + 1
        st.selectbox(
            "Select a Chat ID to view:",
            ["--- Select a chat ---"] + st.session_state.admin_user_chat_ids,
            key="admin_chat_selector",
            index=current_chat_index,
            on_change=_update_admin_chat_selection_callback,
        )
        st.text_input(
            "Or enter Chat ID to view:",
            key="admin_selected_chat_id_input_widget",
            on_change=_update_admin_chat_id_from_input,
        )
        if st.button("View Chat", key="view_chat_button"):
            _sync_admin_inputs()
            if not st.session_state.admin_selected_user_id or not st.session_state.admin_selected_chat_id:
                st.warning("Please select a User ID and a Chat ID.")
            elif st.session_state.admin_selected_chat_id not in st.session_state.admin_user_chat_ids:
                st.error("Selected Chat ID not found for the given User ID.")
            else:
                st.session_state.view_chat = True
                st.rerun()

    st.markdown("--- ")
    if st.button("Back to Role Selection", key="back_to_role_admin"):
        _clear_session_state_for_role_selection()
        st.rerun()
