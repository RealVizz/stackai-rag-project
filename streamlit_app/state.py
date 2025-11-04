import streamlit as st


# --- Session State Initialization ---

def initialize_session_state():
    """
    Initializes all required session state variables for the application.

    This function is called at the beginning of the script run to ensure that all necessary
    keys exist in `st.session_state`. This prevents `AttributeError` exceptions and allows
    for a centralized and predictable state management.

    The state is divided into several categories:
    - General State: For application-wide variables like the user's role.
    - User-Specific State: For data related to a logged-in user, such as their chats and chat history.
    - Admin-Specific State: For data related to the admin dashboard, like the list of all users.
    - Widget State: To hold the values of input widgets. This is part of a strategy to manually
      control and sync widget values with the persistent session state, avoiding certain Streamlit quirks.
    - File Uploader State: To manage the state of the file uploader, including a dynamic key to allow reset.
    """
    # General state
    if "role" not in st.session_state:
        st.session_state.role = None

    # User specific state
    if "user_id" not in st.session_state:
        st.session_state.user_id = ""  # The user's unique identifier.
    if "chat_ids" not in st.session_state:
        st.session_state.chat_ids = []  # List of chat IDs for the current user.
    if "selected_chat_id" not in st.session_state:
        st.session_state.selected_chat_id = ""  # The chat ID currently being viewed or selected.
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # The message history of the selected chat.
    if "new_chat_name" not in st.session_state:
        st.session_state.new_chat_name = ""  # Holds the name for a new chat to be created.
    if "uploaded_files_info" not in st.session_state:
        st.session_state.uploaded_files_info = {}  # Stores info about uploaded files per chat.
    if "user_view_chat" not in st.session_state:
        st.session_state.user_view_chat = False  # Flag to switch to the user chat window.

    # File uploader state
    if "file_uploader_active" not in st.session_state:
        st.session_state.file_uploader_active = False  # True if files are staged in the uploader.
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0  # A key that can be incremented to reset the file_uploader widget.

    # User widget state (for manual control)
    if "user_id_input_widget" not in st.session_state:
        st.session_state.user_id_input_widget = ""
    if "selected_chat_id_input_widget" not in st.session_state:
        st.session_state.selected_chat_id_input_widget = ""
    if "new_chat_name_input_widget" not in st.session_state:
        st.session_state.new_chat_name_input_widget = ""

    # Admin specific state
    if "all_user_ids" not in st.session_state:
        st.session_state.all_user_ids = []  # List of all user IDs in the system.
    if "admin_selected_user_id" not in st.session_state:
        st.session_state.admin_selected_user_id = ""  # The user ID selected by the admin.
    if "admin_user_chat_ids" not in st.session_state:
        st.session_state.admin_user_chat_ids = []  # Chat IDs for the user selected by the admin.
    if "admin_selected_chat_id" not in st.session_state:
        st.session_state.admin_selected_chat_id = ""  # The chat ID selected by the admin.
    if "view_chat" not in st.session_state:
        st.session_state.view_chat = False  # Flag to switch to the admin chat view window.

    # Admin widget state (for manual control)
    if "admin_user_id_input_widget" not in st.session_state:
        st.session_state.admin_user_id_input_widget = ""
    if "admin_selected_chat_id_input_widget" not in st.session_state:
        st.session_state.admin_selected_chat_id_input_widget = ""


# --- State Management Functions (Callbacks & Handlers) ---

def _clear_session_state_for_role_selection():
    """
    Resets the entire session state to its initial default values.
    This is used when navigating back to the main role selection screen to ensure a clean slate.
    It iterates through all keys in the session state, deletes them, and then calls
    `initialize_session_state` again.
    """
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    initialize_session_state()
    st.session_state.role = None  # Explicitly set role to None for the welcome screen.


# --- Callbacks for Selectbox Widgets ---

def _update_user_chat_selection_callback():
    """
    Callback for the user's chat selection dropdown (`st.selectbox`).
    It syncs the value of the dropdown with the main `selected_chat_id` state variable
    and also updates the corresponding text input widget to reflect the selection.
    """
    chat_id = st.session_state.get("chat_selector", "--- Select a chat ---")
    if chat_id == "--- Select a chat ---":
        chat_id = ""
    st.session_state.selected_chat_id = chat_id
    st.session_state.selected_chat_id_input_widget = chat_id


def _update_admin_user_selection_callback():
    """
    Callback for the admin's user selection dropdown.
    It updates the `admin_selected_user_id` and its corresponding text input widget.
    Crucially, it also resets the state for dependent selections (user's chats) to prevent inconsistency.
    """
    user_id = st.session_state.get("admin_user_selector", "--- Select a user ---")
    if user_id == "--- Select a user ---":
        user_id = ""
    st.session_state.admin_selected_user_id = user_id
    st.session_state.admin_user_id_input_widget = user_id
    # Reset dependent state since the selected user has changed.
    st.session_state.admin_user_chat_ids = []
    st.session_state.admin_selected_chat_id = ""
    st.session_state.admin_selected_chat_id_input_widget = ""


def _update_admin_chat_selection_callback():
    """
    Callback for the admin's chat selection dropdown.
    Syncs the selected chat ID with the `admin_selected_chat_id` state and its text input widget.
    """
    chat_id = st.session_state.get("admin_chat_selector", "--- Select a chat ---")
    if chat_id == "--- Select a chat ---":
        chat_id = ""
    st.session_state.admin_selected_chat_id = chat_id
    st.session_state.admin_selected_chat_id_input_widget = chat_id


# --- Callbacks for Text Input Widgets ---
# These functions are used with the `on_change` parameter of `st.text_input`.
# They ensure that any user input into a text field is immediately synced with the
# corresponding variable in the main session state.

def _update_user_id_from_input():
    st.session_state.user_id = st.session_state.get("user_id_input_widget", "")


def _update_chat_id_from_input():
    st.session_state.selected_chat_id = st.session_state.get("selected_chat_id_input_widget", "")


def _update_new_chat_name_from_input():
    st.session_state.new_chat_name = st.session_state.get("new_chat_name_input_widget", "")


def _update_admin_user_id_from_input():
    st.session_state.admin_selected_user_id = st.session_state.get("admin_user_id_input_widget", "")


def _update_admin_chat_id_from_input():
    st.session_state.admin_selected_chat_id = st.session_state.get("admin_selected_chat_id_input_widget", "")


# --- File Uploader Callbacks & Handlers ---

def _file_uploader_on_change_callback():
    """
    Callback for the `st.file_uploader` widget.
    It sets a flag (`file_uploader_active`) to True if files have been staged in the uploader.
    This flag can be used to disable other UI elements, like the chat input, to prevent
    conflicting actions.
    """
    current_key = f"pdf_uploader_{st.session_state.uploader_key}"
    st.session_state.file_uploader_active = bool(st.session_state.get(current_key))


def _handle_file_upload():
    """
    Handler for the "Upload Selected Files" button's `on_click` event.
    It retrieves the staged files and calls the backend utility function to perform the upload.
    After the upload, it resets the uploader by incrementing its dynamic key, ensuring the
    widget is cleared and ready for new files.
    """
    user_id = st.session_state.user_id
    chat_id = st.session_state.selected_chat_id
    current_key = f"pdf_uploader_{st.session_state.uploader_key}"
    files_to_upload = st.session_state.get(current_key, [])

    if not user_id or not chat_id:
        st.warning("Cannot upload: missing User ID or Chat ID.")
        return
    if not files_to_upload:
        st.warning("No files selected to upload.")
        return

    with st.spinner("Uploading and processing files..."):
        # This function is defined in `utils.py` and handles the HTTP request.
        from streamlit_app.utils import upload_pdf_to_backend
        upload_results = upload_pdf_to_backend(user_id, chat_id, files_to_upload)

    if chat_id not in st.session_state.uploaded_files_info:
        st.session_state.uploaded_files_info[chat_id] = []
    st.session_state.uploaded_files_info[chat_id].extend(upload_results)
    st.success("File processing complete!")

    # Reset the file uploader by changing its key and resetting the active flag.
    st.session_state.uploader_key += 1
    st.session_state.file_uploader_active = False


# --- Manual Sync Functions for Button Clicks ---
# These functions are called when a button is pressed to ensure that the latest values
# from all relevant text input widgets are synced to the main session state *before*
# the button's primary action is executed. This is a workaround for Streamlit's execution model
# where widget values are not immediately available in the script run of a button press.

def _sync_user_inputs():
    """Manually syncs all inputs from the user dashboard to the session state."""
    _update_user_id_from_input()
    _update_chat_id_from_input()
    _update_new_chat_name_from_input()


def _sync_admin_inputs():
    """Manually syncs all inputs from the admin dashboard to the session state."""
    _update_admin_user_id_from_input()
    _update_admin_chat_id_from_input()
