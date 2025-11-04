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
    get_all_user_ids
)


# --- Page: Welcome ---

def page_welcome():
    """
    Renders the initial role selection screen for the application.
    This is the first page the user sees. It allows them to choose between being a 'User' or an 'Admin'.
    The choice updates the `role` in the session state, and a `st.rerun()` is called to immediately
    trigger the main app router in `app.py` to display the correct dashboard.
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
    Renders the main chat interface.
    This view is used by both regular users and admins (in a read-only mode).

    Args:
        is_admin_view: A boolean flag to determine if the view is for an admin.
                       If True, chat input and file uploading are disabled.
    """
    # Determine the correct user and chat ID based on the view type.
    user_id = st.session_state.user_id if not is_admin_view else st.session_state.admin_selected_user_id
    chat_id = st.session_state.selected_chat_id if not is_admin_view else st.session_state.admin_selected_chat_id

    # A crucial validation check. If there's no user or chat ID, display a warning and a back button.
    if not user_id or not chat_id:
        st.warning("Please select a User ID and Chat ID to view messages.")
        if st.button("Go Back"):
            # Reset the appropriate state variables to exit the chat view.
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

    # Load and display the chat history from the session state.
    if not st.session_state.chat_history:
        st.session_state.chat_history = get_chat_history(user_id, chat_id)

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Display the list of uploaded files for the current chat in the sidebar.
    if not is_admin_view and st.session_state.uploaded_files_info.get(chat_id):
        st.sidebar.subheader("Uploaded Files")
        for file_info in st.session_state.uploaded_files_info[chat_id]:
            status_icon = "✅" if file_info.get("status") == "success" else "❌"
            st.sidebar.write(f'{status_icon} {file_info.get("filename", "")}')

    # --- User-specific components (Chat Input & File Uploader) ---
    if not is_admin_view:
        # The main chat input box.
        user_query = st.chat_input(
            "Type your message here...",
            key="chat_input",
            disabled=st.session_state.file_uploader_active,  # Disable if files are staged for upload.
        )
        if st.session_state.file_uploader_active and not user_query:
            st.info("Files are ready to upload in the sidebar. Upload them or remove them to continue chatting.")

        # If the user enters a query, process it.
        if user_query:
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.spinner("Thinking..."):
                response = query_backend(user_id, chat_id, user_query)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()  # Rerun to display the new messages.

        # The file uploader is placed in the sidebar.
        with st.sidebar:
            st.subheader("Upload PDF Files")
            # A dynamic key is used to allow the uploader to be reset programmatically.
            current_uploader_key = f"pdf_uploader_{st.session_state.uploader_key}"

            st.file_uploader(
                "Choose PDF files",
                type="pdf",
                accept_multiple_files=True,
                key=current_uploader_key,
                on_change=_file_uploader_on_change_callback,
            )

            can_upload = bool(user_id and chat_id)
            if not can_upload:
                st.info("Enter your User ID and select (or create) a Chat ID before uploading files.")

            # The upload button is disabled until files are staged and IDs are present.
            upload_button_disabled = not (st.session_state.file_uploader_active and can_upload)
            st.button(
                "Upload Selected Files",
                key="upload_button",
                disabled=upload_button_disabled,
                on_click=_handle_file_upload
            )
    else:
        # For admins, the view is read-only.
        st.info("Admin View: You can only view this chat. Interaction is disabled.")
        st.chat_input("Admin view: Chat is disabled.", disabled=True)


# --- Page: User Dashboard ---

def page_user_dashboard():
    """
    Renders the main dashboard for a regular user.
    This page allows a user to manage their chats: fetch existing ones, select one to enter,
    or create a new one.
    """
    st.header("User Dashboard")

    # --- User ID Input ---
    st.text_input(
        "Enter your User ID:",
        key="user_id_input_widget",
        on_change=_update_user_id_from_input,
    )

    # --- Fetch Chats Button ---
    if st.button("Fetch My Chats", key="fetch_chats_button"):
        _sync_user_inputs()  # Sync inputs before processing.
        if st.session_state.user_id:
            with st.spinner("Fetching chats..."):
                st.session_state.chat_ids = get_all_chat_ids_for_user(st.session_state.user_id)
            if not st.session_state.chat_ids:
                st.info("You don't have any chats yet. Start a new one!")
            else:
                st.success(f"Found {len(st.session_state.chat_ids)} chats.")
        else:
            st.warning("Please enter a User ID.")

    # --- Existing Chat Selector ---
    # This section is shown only if there are chats to select from.
    if st.session_state.chat_ids or st.session_state.selected_chat_id:
        # Determine the index for the selectbox to keep it in sync with the state.
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
        # This button is the sole entry point into an existing chat.
        if st.button("Enter Chat", key="enter_chat_button"):
            _sync_user_inputs()
            if not st.session_state.user_id:
                st.warning("Please enter a User ID.")
            elif not st.session_state.selected_chat_id:
                st.warning("Please select or enter a Chat ID.")
            elif st.session_state.selected_chat_id not in st.session_state.chat_ids:
                st.error(f"Chat ID '{st.session_state.selected_chat_id}' not found for user. Please fetch chats first.")
            else:
                # Set the flag to true to switch to the chat view.
                st.session_state.user_view_chat = True
                st.rerun()

    # --- New Chat Form ---
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
                # Add the new chat to the state and switch to the chat view.
                new_chat_name = st.session_state.new_chat_name
                st.session_state.selected_chat_id = new_chat_name
                st.session_state.chat_ids.append(new_chat_name)
                st.session_state.chat_history = []
                st.session_state.new_chat_name = ""  # Clear input field
                st.session_state.new_chat_name_input_widget = ""
                st.session_state.user_view_chat = True
                st.rerun()
        else:
            st.warning("Please enter both a User ID and a New Chat Name.")

    # --- Back Button ---
    st.markdown("--- ")
    if st.button("Back to Role Selection", key="back_to_role_user"):
        _clear_session_state_for_role_selection()
        st.rerun()


# --- Page: Admin Dashboard ---

def page_admin_dashboard():
    """Renders the dashboard for the admin to view user chats."""
    st.header("Admin Dashboard")

    # --- Admin User Selector ---
    fetch_users_col, user_dropdown_col = st.columns([0.3, 0.7])
    with fetch_users_col:
        if st.button("Fetch All Users", key="fetch_all_users_admin", use_container_width=True):
            with st.spinner("Fetching all users..."):
                st.session_state.all_user_ids = get_all_user_ids()
            if not st.session_state.all_user_ids:
                st.info("No users found yet.")
            else:
                st.success(f"Found {len(st.session_state.all_user_ids)} users.")
                # Reset selection if the current one is no longer valid.
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

    # --- Admin Chat Selector ---
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

    # --- Back Button ---
    st.markdown("--- ")
    if st.button("Back to Role Selection", key="back_to_role_admin"):
        _clear_session_state_for_role_selection()
        st.rerun()
