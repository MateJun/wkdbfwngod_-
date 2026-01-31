import win32com.client

def find_video_device_ids():
    """
    시스템에 연결된 비디오 캡처 장치의 이름과 고유 Device ID를 찾아서 출력합니다.
    """
    wmi = win32com.client.GetObject("winmgmts:")
    devices = wmi.InstancesOf("Win32_PnPEntity")
    
    print("=" * 70)
    print("시스템에 연결된 비디오 캡처 장치 목록:")
    print("=" * 70)
    
    video_devices = []
    for device in devices:
        # 'Video' 또는 'Camera' 카테고리에 속하는 장치들을 찾습니다.
        if device.PNPClass in ('Image', 'Camera', 'Media'):
            try:
                # 장치 이름(Caption)과 Device ID를 가져옵니다.
                device_name = device.Caption
                device_id = device.DeviceID
                
                # 비디오 캡처와 관련된 장치일 가능성이 높은 것들만 필터링
                if 'video' in device_name.lower() or 'camera' in device_name.lower() or 'webcam' in device_name.lower():
                    video_devices.append({'name': device_name, 'id': device_id})
                    print(f"  - 이름 (Name): {device_name}")
                    print(f"    ID (DeviceID): {device_id}\n")
            except Exception:
                pass
                
    if not video_devices:
        print("비디오 캡처 장치를 찾을 수 없습니다.")
        
    print("=" * 70)
    print("위 목록에서 'Logitech HD Pro Webcam C920'에 해당하는 두 개의 'ID'를 복사하여 알려주세요.")
    print("이 ID를 사용하여 왼쪽/오른쪽 카메라를 영구적으로 구별하게 됩니다.")

if __name__ == "__main__":
    find_video_device_ids()
