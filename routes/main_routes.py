from flask import Blueprint, render_template, send_from_directory
from config import RESULT_FOLDER, TEMPLATE_FOLDER

# Blueprint를 생성할 때 template_folder를 지정해야 합니다.
main_bp = Blueprint('main_bp', __name__, template_folder=TEMPLATE_FOLDER)

@main_bp.route("/")
def index():
    """메인 index.html 페이지를 렌더링합니다."""
    return render_template('index.html')

@main_bp.route('/results/<filename>')
def serve_result(filename):
    """'results' 폴더에 있는 처리된 파일을 제공합니다."""
    return send_from_directory(RESULT_FOLDER, filename)
