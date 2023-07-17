set user_services wireplumber.service pipewire.service plasma-kded.service plasma-kwin_x11 plasma-plasmashell plasma-xdg-desktop-portal-kde.service
set system_services NetworkManager avahi-daemon bluetooth polkit upower rtkit-daemon sddm

mkdir -p user
for svc in $user_services
   journalctl -u $svc --since="1970-01-01" -o json- --user &> user/(basename $svc ".service").json
end

mkdir -p system
for svc in $system_services
   journalctl -u $svc --since="1970-01-01" -o json &> system/(basename $svc ".service").json
end
